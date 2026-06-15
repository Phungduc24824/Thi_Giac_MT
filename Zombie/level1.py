import cv2
import pygame
import random
import os
import sys
import urllib.request
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat, Image as MediaImage

# ================== DOWNLOAD MODEL ==================
def download_hand_landmarker_model():

    possible_paths = [
        'C:\\temp\\hand_landmarker.task',
        'hand_landmarker.task'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Using model from: {path}")
            return path

    model_path = 'C:\\temp\\hand_landmarker.task'

    os.makedirs('C:\\temp', exist_ok=True)

    print("Downloading model...")

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    try:
        urllib.request.urlretrieve(url, model_path)
        print("✓ Download complete")
        return model_path

    except Exception as e:
        print("Download failed:", e)
        return None


# ================== INIT ==================
pygame.init()
pygame.mixer.init()

WIDTH = 1550
HEIGHT = 800

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Zombie Hand Shooter - Level 1")

clock = pygame.time.Clock()

# ================== COLORS ==================
BLACK = (15, 15, 15)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (180, 0, 255)

# ================== BACKGROUND ==================
if not os.path.exists("assets/city_1.jpg"):
    print("❌ Missing assets/city_1.jpg")
    sys.exit()

background = pygame.image.load(
    "assets/city_1.jpg"
).convert()

background = pygame.transform.scale(
    background,
    (WIDTH, HEIGHT)
)

# ================== LOAD SOUNDS ==================
gun_sound = None
hit_sound = None

if os.path.exists("assets/gun.wav"):
    gun_sound = pygame.mixer.Sound("assets/gun.wav")
else:
    print("⚠️ Missing sound: assets/gun.wav")

if os.path.exists("assets/hit.wav"):
    hit_sound = pygame.mixer.Sound("assets/hit.wav")
else:
    print("⚠️ Missing sound: assets/hit.wav")

# ================== LOAD ZOMBIE ==================
if not os.path.exists("assets/zombies.png"):
    print("❌ Missing zombies.png")
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

# ================== HAND TRACKING ==================
model_path = download_hand_landmarker_model()

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

print("✓ Hand tracking ready")

# ================== CAMERA ==================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():

    print("❌ Cannot open camera")

    pygame.quit()
    sys.exit()

print("✓ Camera ready")

# ================== PARTICLE ==================
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

# ================== HAND GESTURE ==================
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

# ================== PLAYER DATA ==================
score = 0

target_kill = 10

shoot_cooldown = 0

player_hp = 100
max_hp = 100

virus_infected = False
virus_timer = 0

# ================== ZOMBIE ==================
zombies = []

class Zombie:

    def __init__(self):

        self.infected = False

        self.respawn()

    def respawn(self):

        valid_position = False

        while not valid_position:

            new_x = random.randint(WIDTH + 100, WIDTH + 600)

            new_y = random.randint(120, HEIGHT - 120)

            valid_position = True

            for z in zombies:

                distance_x = abs(new_x - z.x)
                distance_y = abs(new_y - z.y)

                if distance_x < 180 and distance_y < 120:
                    valid_position = False
                    break

        self.x = new_x
        self.y = new_y

        self.speed = random.uniform(2, 5)

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

# ================== CREATE ZOMBIES ==================
TOTAL_ZOMBIES = 8
MAX_INFECTED = 2

infected_count = 0

for _ in range(TOTAL_ZOMBIES):

    z = Zombie()

    if infected_count < MAX_INFECTED:

        z.infected = True

        infected_count += 1

    zombies.append(z)

# ================== CURSOR ==================
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

level2_btn = Button(
    WIDTH // 2 + 50,
    HEIGHT // 2 + 80,
    220,
    80,
    "LEVEL 2",
    (170, 0, 0)
)

# ================== MAIN LOOP ==================
running = True

while running:

    screen.blit(background, (0, 0))

    # ================== VIRUS DAMAGE ==================
    if virus_infected:

        virus_timer -= 1

        if virus_timer % 60 == 0:

            player_hp -= 1

            if player_hp < 0:
                player_hp = 0

        if virus_timer <= 0:
            virus_infected = False

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    media_image = MediaImage(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(media_image)

    action = "NONE"

    # ================== HAND DETECT ==================
    if result.hand_landmarks:

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

    # ================== SHOOT ==================
    if shoot_cooldown > 0:
        shoot_cooldown -= 1

    if action == "SHOOT" and shoot_cooldown == 0:

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

                    print("💀 Zombie killed!")

    # ================== UPDATE ==================
    for z in zombies:
        z.move()

    # ================== DRAW ==================
    for z in zombies:
        z.draw()

    # ================== BLOOD ==================
    for particle in blood_particles[:]:

        particle.update()

        particle.draw()

        if particle.life <= 0:
            blood_particles.remove(particle)

    # ================== CROSSHAIR ==================
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

    # ================== UI ==================
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

    # ================== HP ==================
    bar_x = 20
    bar_y = 120

    bar_width = 300
    bar_height = 30

    pygame.draw.rect(
        screen,
        RED,
        (bar_x, bar_y, bar_width, bar_height)
    )

    pygame.draw.rect(
        screen,
        GREEN,
        (
            bar_x,
            bar_y,
            bar_width * (player_hp / max_hp),
            bar_height
        )
    )

    pygame.draw.rect(
        screen,
        WHITE,
        (bar_x, bar_y, bar_width, bar_height),
        3
    )

    hp_text = font.render(
        f"HP: {player_hp}/100",
        True,
        WHITE
    )

    screen.blit(hp_text, (340, 115))

    # ================== INFECTED ==================
    if virus_infected:

        virus_text = font.render(
            "INFECTED!",
            True,
            PURPLE
        )

        screen.blit(virus_text, (20, 170))

    # ================== GAME OVER ==================
    if player_hp <= 0:

        over_font = pygame.font.SysFont(None, 100)

        game_over_text = over_font.render(
            "GAME OVER",
            True,
            RED
        )

        screen.blit(
            game_over_text,
            (WIDTH // 2 - 250, HEIGHT // 2 - 50)
        )

        pygame.display.flip()

        pygame.time.delay(3000)

        running = False

    # ================== LEVEL COMPLETE ==================
    if score >= target_kill:

        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))

        screen.blit(overlay, (0, 0))

        win_font = pygame.font.SysFont(None, 100)

        win_text = win_font.render(
            "LEVEL 1 COMPLETE",
            True,
            GREEN
        )

        screen.blit(
            win_text,
            (WIDTH // 2 - 320, HEIGHT // 2 - 120)
        )

        play_again_btn.draw()
        level2_btn.draw()

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

                        python = sys.executable

                        os.execl(
                            python,
                            python,
                            *sys.argv
                        )

                    # LEVEL 2
                    if level2_btn.is_clicked(mouse_pos):

                        print("➡ Open Level 2")

                        waiting = False
                        running = False

    # ================== HAND CAMERA ==================
    small_frame = cv2.resize(frame, (100, 80))

    cv2.putText(
        small_frame,
        action,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1
    )

    window_name = "HandCam"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        150,
        100
    )

    screen_info = pygame.display.Info()

    screen_h = screen_info.current_h

    cam_x = 0
    cam_y = screen_h - 120

    cv2.moveWindow(
        window_name,
        cam_x,
        cam_y
    )

    cv2.imshow(
        window_name,
        small_frame
    )

    pygame.display.flip()

    clock.tick(40)

    # ================== EVENTS ==================
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    key = cv2.waitKey(1)

    if key == 27:
        break

# ================== CLEAN ==================
cap.release()

cv2.destroyAllWindows()

pygame.quit()