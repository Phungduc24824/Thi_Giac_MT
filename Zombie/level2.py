import cv2
import pygame
import random
import os
import sys
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat, Image as MediaImage

# ================= DOWNLOAD MODEL =================

def download_hand_landmarker_model():

    possible_paths = [
        "C:/temp/hand_landmarker.task",
        "hand_landmarker.task"
    ]

    for path in possible_paths:

        if os.path.exists(path):
            print("✓ Using model:", path)
            return path

    os.makedirs("C:/temp", exist_ok=True)

    model_path = "C:/temp/hand_landmarker.task"

    url = (
        "https://storage.googleapis.com/"
        "mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/"
        "hand_landmarker.task"
    )

    print("Downloading model...")

    urllib.request.urlretrieve(url, model_path)

    print("✓ Download complete")

    return model_path

# ================= INIT =================

pygame.init()
pygame.mixer.init()

WIDTH = 1550
HEIGHT = 800

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Zombie Hand Shooter - Level 2")

clock = pygame.time.Clock()

# ================= COLORS =================

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 150, 255)
YELLOW = (255, 255, 0)
PURPLE = (180, 0, 255)

# ================= BACKGROUND =================

if not os.path.exists("assets/city_2.jpg"):

    print("❌ Missing assets/city_2.jpg")
    sys.exit()

background = pygame.image.load(
    "assets/city_2.jpg"
).convert()

background = pygame.transform.scale(
    background,
    (WIDTH, HEIGHT)
)

# ================= SOUNDS =================

gun_sound = None
hit_sound = None

if os.path.exists("assets/gun.wav"):
    gun_sound = pygame.mixer.Sound("assets/gun.wav")

if os.path.exists("assets/hit.wav"):
    hit_sound = pygame.mixer.Sound("assets/hit.wav")

# ================= LOAD ZOMBIE =================

if not os.path.exists("assets/zombies.png"):

    print("❌ Missing assets/zombies.png")
    sys.exit()

sprite_sheet = pygame.image.load(
    "assets/zombies.png"
).convert_alpha()

sheet_width = sprite_sheet.get_width()
sheet_height = sprite_sheet.get_height()

frame_width = sheet_width // 4
frame_height = sheet_height // 2

zombie_img = sprite_sheet.subsurface(
    (0, 0, frame_width, frame_height)
)

zombie_img = pygame.transform.scale(
    zombie_img,
    (120, 160)
)

zombie_img = pygame.transform.flip(
    zombie_img,
    True,
    False
)

# ================= HAND TRACKING =================

model_path = download_hand_landmarker_model()

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)

print("✓ Hand tracking ready")

# ================= CAMERA =================

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():

    print("❌ Cannot open camera")
    pygame.quit()
    sys.exit()

print("✓ Camera ready")

# ================= BLOOD EFFECT =================

blood_particles = []

class BloodParticle:

    def __init__(self, x, y):

        self.x = x
        self.y = y

        self.size = random.randint(3, 7)

        self.speed_x = random.uniform(-5, 5)
        self.speed_y = random.uniform(-5, 5)

        self.life = 30

    def update(self):

        self.x += self.speed_x
        self.y += self.speed_y

        self.life -= 1

    def draw(self):

        pygame.draw.circle(
            screen,
            RED,
            (int(self.x), int(self.y)),
            self.size
        )

# ================== BUTTON ==================
class Button:

    def __init__(self, x, y, width, height, text, color):

        self.rect = pygame.Rect(x, y, width, height)

        self.text = text

        self.color = color

    def draw(self):

        pygame.draw.rect(
            screen,
            self.color,
            self.rect,
            border_radius=12
        )

        pygame.draw.rect(
            screen,
            WHITE,
            self.rect,
            3,
            border_radius=12
        )

        font = pygame.font.SysFont(None, 45)

        text_surface = font.render(
            self.text,
            True,
            WHITE
        )

        text_rect = text_surface.get_rect(
            center=self.rect.center
        )

        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos):

        return self.rect.collidepoint(pos)

# ================= GESTURE DETECTION =================

def detect_action(landmarks):

    if not landmarks:
        return "NONE"

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]

    # AIM

    index_up = index_tip.y < index_pip.y

    middle_down = middle_tip.y > middle_pip.y
    ring_down = ring_tip.y > ring_pip.y
    pinky_down = pinky_tip.y > pinky_pip.y

    if index_up and middle_down and ring_down and pinky_down:
        return "AIM"

    # CHARGE

    open_count = 0

    if index_tip.y < index_pip.y:
        open_count += 1

    if middle_tip.y < middle_pip.y:
        open_count += 1

    if ring_tip.y < ring_pip.y:
        open_count += 1

    if pinky_tip.y < pinky_pip.y:
        open_count += 1

    if open_count >= 3:
        return "CHARGE"

    # SHOOT

    fist_closed = (
        index_tip.y > index_pip.y and
        middle_tip.y > middle_pip.y and
        ring_tip.y > ring_pip.y and
        pinky_tip.y > pinky_pip.y
    )

    if fist_closed:
        return "SHOOT"

    return "NONE"

# ================= PLAYER =================

score = 0
target_kill = 30

shoot_cooldown = 0

player_hp = 100
max_hp = 100

energy = 0
max_energy = 100

charging = False

virus_infected = False
virus_timer = 0

# ================= ZOMBIE =================

zombies = []

class Zombie:

    def __init__(self):

        self.infected = random.random() < 0.3

        self.respawn()

    def respawn(self):

        self.x = random.randint(
            WIDTH + 100,
            WIDTH + 600
        )

        self.y = random.randint(
            120,
            HEIGHT - 120
        )

        if self.infected:

            self.speed = random.uniform(2.5, 5)

        else:

            self.speed = random.uniform(1.5, 4)

        self.hp = 100

    def move(self):

        global player_hp
        global virus_infected
        global virus_timer

        self.x -= self.speed

        self.y += random.randint(-1, 1)

        if self.x < -150:

            player_hp -= 10

            if player_hp < 0:
                player_hp = 0

            if self.infected:

                virus_infected = True
                virus_timer = 600

            self.respawn()

    def draw(self):

        if self.infected:

            pygame.draw.circle(
                screen,
                PURPLE,
                (int(self.x), int(self.y)),
                70,
                4
            )

        screen.blit(
            zombie_img,
            (self.x - 60, self.y - 80)
        )

        pygame.draw.rect(
            screen,
            RED,
            (self.x - 40, self.y - 90, 80, 8)
        )

        pygame.draw.rect(
            screen,
            GREEN,
            (
                self.x - 40,
                self.y - 90,
                80 * (self.hp / 100),
                8
            )
        )

# ================= CREATE ZOMBIES =================

for _ in range(8):

    zombies.append(Zombie())

# ================= CURSOR =================

hand_x = WIDTH // 2
hand_y = HEIGHT // 2

# ================== BUTTONS ==================
play_again_btn = Button(
    WIDTH // 2 - 270,
    HEIGHT // 2 + 80,
    220,
    80,
    "PLAY AGAIN",
    (0, 170, 0)
)

level3_btn = Button(
    WIDTH // 2 + 50,
    HEIGHT // 2 + 80,
    220,
    80,
    "LEVEL 3",
    (170, 0, 0)
)

# ================= MAIN LOOP =================

frame_skip = 0

running = True

while running:

    screen.blit(background, (0, 0))

    # EVENTS

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_q]:
        running = False

    # CAMERA

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    media_image = MediaImage(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    frame_skip += 1

    result = None

    if frame_skip % 2 == 0:
        result = landmarker.detect(media_image)

    action = "NONE"

    # HAND DETECT

    if result and result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        action = detect_action(landmarks)

        target_x = int(landmarks[8].x * WIDTH)
        target_y = int(landmarks[8].y * HEIGHT)

        hand_x = int(hand_x * 0.7 + target_x * 0.3)
        hand_y = int(hand_y * 0.7 + target_y * 0.3)

        for lm in landmarks:

            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

    cursor = (hand_x, hand_y)

    # ================= ENERGY =================

    if action == "CHARGE":

        charging = True

        energy += 100 / (25 * 3)

        if energy > 100:
            energy = 100

    else:

        charging = False

    # ================= SPECIAL =================

    if action == "SHOOT" and energy >= 100:

        energy = 0

        pygame.draw.rect(
            screen,
            BLUE,
            (
                cursor[0] - 250,
                cursor[1] - 250,
                500,
                500
            ),
            10
        )

        for z in zombies:

            if (
                abs(z.x - cursor[0]) < 250 and
                abs(z.y - cursor[1]) < 250
            ):

                for _ in range(40):

                    blood_particles.append(
                        BloodParticle(z.x, z.y)
                    )

                z.respawn()

                score += 1

    # ================= NORMAL SHOOT =================

    if shoot_cooldown > 0:
        shoot_cooldown -= 1

    if (
        action == "SHOOT"
        and shoot_cooldown == 0
        and energy < 100
    ):

        shoot_cooldown = 12

        if gun_sound:
            gun_sound.play()

        pygame.draw.circle(
            screen,
            WHITE,
            cursor,
            35,
            5
        )

        for z in zombies:

            dx = abs(cursor[0] - z.x)
            dy = abs(cursor[1] - z.y)

            if dx < 60 and dy < 60:

                z.hp -= 50

                if hit_sound:
                    hit_sound.play()

                for _ in range(20):

                    blood_particles.append(
                        BloodParticle(z.x, z.y)
                    )

                if z.hp <= 0:

                    z.respawn()

                    score += 1

    # ================= VIRUS DAMAGE =================

    if virus_infected:

        virus_timer -= 1

        if virus_timer % 60 == 0:

            player_hp -= 1

        if virus_timer <= 0:

            virus_infected = False

    # ================= UPDATE =================

    for z in zombies:
        z.move()

    # ================= DRAW =================

    for z in zombies:
        z.draw()

    # ================= BLOOD =================

    for particle in blood_particles[:]:

        particle.update()
        particle.draw()

        if particle.life <= 0:
            blood_particles.remove(particle)

    # ================= CROSSHAIR =================

    pygame.draw.circle(
        screen,
        RED,
        cursor,
        15,
        2
    )

    pygame.draw.line(
        screen,
        RED,
        (cursor[0] - 20, cursor[1]),
        (cursor[0] + 20, cursor[1]),
        2
    )

    pygame.draw.line(
        screen,
        RED,
        (cursor[0], cursor[1] - 20),
        (cursor[0], cursor[1] + 20),
        2
    )

    # ================= UI =================

    font = pygame.font.SysFont(None, 40)

    score_text = font.render(
        f"KILLS: {score}/{target_kill}",
        True,
        WHITE
    )

    screen.blit(score_text, (20, 20))

    action_text = font.render(
        f"ACTION: {action}",
        True,
        YELLOW
    )

    screen.blit(action_text, (20, 70))

    # HP

    pygame.draw.rect(
        screen,
        RED,
        (20, 120, 300, 30)
    )

    pygame.draw.rect(
        screen,
        GREEN,
        (
            20,
            120,
            300 * (player_hp / max_hp),
            30
        )
    )

    pygame.draw.rect(
        screen,
        WHITE,
        (20, 120, 300, 30),
        3
    )

    hp_text = font.render(
        f"HP: {player_hp}/100",
        True,
        WHITE
    )

    screen.blit(hp_text, (340, 115))

    # ENERGY

    pygame.draw.rect(
        screen,
        BLACK,
        (20, 170, 300, 25)
    )

    pygame.draw.rect(
        screen,
        BLUE,
        (
            20,
            170,
            300 * (energy / max_energy),
            25
        )
    )

    pygame.draw.rect(
        screen,
        WHITE,
        (20, 170, 300, 25),
        3
    )

    energy_text = font.render(
        f"ENERGY: {int(energy)}%",
        True,
        WHITE
    )

    screen.blit(energy_text, (340, 165))

    # STATUS

    if charging:

        charge_text = font.render(
            "CHARGING...",
            True,
            BLUE
        )

        screen.blit(charge_text, (20, 210))

    if energy >= 100:

        ready_text = font.render(
            "SPECIAL READY!",
            True,
            BLUE
        )

        screen.blit(ready_text, (20, 250))

    if virus_infected:

        virus_text = font.render(
            "INFECTED!",
            True,
            PURPLE
        )

        screen.blit(virus_text, (20, 290))

    # ================= GAME OVER =================

    if player_hp <= 0:

        over_font = pygame.font.SysFont(None, 100)

        game_over_text = over_font.render(
            "GAME OVER",
            True,
            RED
        )

        screen.blit(
            game_over_text,
            (WIDTH // 2 - 250, HEIGHT // 2)
        )

    # ================= LEVEL COMPLETE =================

    if score >= target_kill:

        overlay = pygame.Surface((WIDTH, HEIGHT))

        overlay.set_alpha(180)

        overlay.fill((0, 0, 0))

        screen.blit(overlay, (0, 0))

        big_font = pygame.font.SysFont(None, 100)

        complete_text = big_font.render(
            "LEVEL 2 COMPLETE",
            True,
            GREEN
        )

        screen.blit(
            complete_text,
            (WIDTH // 2 - 320, HEIGHT // 2 - 120)
        )

        play_again_btn.draw()
        level3_btn.draw()

        pygame.display.flip()

        waiting = True

        while waiting:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    waiting = False
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:

                    mouse_pos = pygame.mouse.get_pos()

                    # PLAY AGAIN
                    if play_again_btn.is_clicked(mouse_pos):

                        cap.release()

                        cv2.destroyAllWindows()

                        pygame.quit()

                        os.system("python level2.py")

                    # LEVEL 3
                    
                    if level3_btn.is_clicked(mouse_pos):

                        cap.release()

                        cv2.destroyAllWindows()

                        pygame.quit()

                        python_exe = sys.executable

                        os.execl(
                            python_exe,
                            python_exe,
                            "level3.py"
                        )

    clock.tick(30)

    # ================= CAMERA WINDOW =================

    small_frame = cv2.resize(
        frame,
        (140, 100)
    )

    cv2.putText(
        small_frame,
        action,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    cv2.imshow(
        "HandCam",
        small_frame
    )

    # ================= UPDATE =================

    pygame.display.flip()

    clock.tick(25)

    if cv2.waitKey(1) == 27:
        running = False

# ================= CLEAN =================

cap.release()

cv2.destroyAllWindows()

pygame.quit()