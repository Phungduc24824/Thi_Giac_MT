# ================= ZOMBIE HAND SHOOTER LEVEL 3 =================

import cv2
import pygame
import random
import os
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

pygame.display.set_caption("Zombie Hand Shooter Level 3")

clock = pygame.time.Clock()

# ================= COLORS =================

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 150, 255)
YELLOW = (255, 255, 0)
PURPLE = (180, 0, 255)

# ================= CHECK ASSETS =================

if not os.path.exists("assets/city_3.jpg"):
    print("❌ Missing assets/city_3.jpg")
    exit()

if not os.path.exists("assets/zombies.png"):
    print("❌ Missing assets/zombies.png")
    exit()

# ================= BACKGROUND =================

background = pygame.image.load(
    "assets/city_3.jpg"
).convert()

background = pygame.transform.scale(
    background,
    (WIDTH, HEIGHT)
)

# ================= SOUND =================

gun_sound = None
hit_sound = None

if os.path.exists("assets/gun.wav"):
    gun_sound = pygame.mixer.Sound("assets/gun.wav")

if os.path.exists("assets/hit.wav"):
    hit_sound = pygame.mixer.Sound("assets/hit.wav")

# ================= LOAD ZOMBIE =================

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

# ================= BOSS IMAGE =================

boss_img = pygame.transform.scale(
    zombie_img,
    (260, 340)
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
    exit()

print("✓ Camera ready")

# ================= PARTICLES =================

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

# ================= GESTURE =================

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

    # 👉 AIM

    index_up = index_tip.y < index_pip.y

    middle_down = middle_tip.y > middle_pip.y
    ring_down = ring_tip.y > ring_pip.y
    pinky_down = pinky_tip.y > pinky_pip.y

    if index_up and middle_down and ring_down and pinky_down:
        return "AIM"

    # ✋ CHARGE

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

    # ✊ SHOOT

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

player_hp = 100
max_hp = 100

shoot_cooldown = 0

energy = 0
max_energy = 100

special_ready = False

# ================= ENERGY BULLET =================

energy_bullets = []

class EnergyBullet:

    def __init__(self, x, y):

        self.x = x
        self.y = y

        self.speed = 25

        self.radius = 20

        self.active = True

    def move(self):

        self.x += self.speed

        if self.x > WIDTH + 100:
            self.active = False

    def draw(self):

        pygame.draw.circle(
            screen,
            BLUE,
            (int(self.x), int(self.y)),
            self.radius
        )

        pygame.draw.circle(
            screen,
            WHITE,
            (int(self.x), int(self.y)),
            self.radius,
            4
        )

# ================= ZOMBIE =================

zombies = []

class Zombie:

    def __init__(self):

        self.hp = 100

        self.respawn()

    def respawn(self):

        self.x = random.randint(
            WIDTH + 100,
            WIDTH + 500
        )

        self.y = random.randint(
            120,
            HEIGHT - 120
        )

        self.speed = random.uniform(2, 3)

        self.hp = 100

    def move(self):

        global player_hp

        self.x -= self.speed

        self.y += random.randint(-1, 1)

        if self.x < -100:

            player_hp -= 10

            self.respawn()

    def draw(self):

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

# ================= BOSS =================

boss_spawned = False
boss_spawn_timer = 0

class BossZombie:

    def __init__(self):

        self.x = WIDTH + 400
        self.y = HEIGHT // 2

        self.hp = 1500
        self.max_hp = 1500

        # tốc độ ban đầu
        self.speed = 1.3

        # phase 2
        self.rage_mode = False

        # phase 3
        self.phase3 = False

        # đã summon chưa
        self.spawned_minions = False

        # hướng né đạn
        self.dodge_dir = 1

    def move(self):

        global player_hp

        # ================= SUMMON MINIONS =================

        if self.hp <= 1000 and not self.spawned_minions:

            self.spawned_minions = True

            # tạo thêm 5 zombie
            for _ in range(5):

                z = Zombie()

                z.x = self.x + random.randint(100, 400)
                z.y = self.y + random.randint(-200, 200)

                # zombie nhanh hơn
                z.speed = random.uniform(3, 6)

                zombies.append(z)

        # ================= PHASE 2 =================

        if self.hp <= 1000 and not self.rage_mode:

            self.rage_mode = True

            # boss nhanh hơn
            self.speed = 1.5

        # ================= PHASE 3 =================

        if self.hp <= 500 and not self.phase3:

            self.phase3 = True

            # cực nhanh
            self.speed = 1.8

            # triệu hồi thêm zombie
            for _ in range(10):

                z = Zombie()

                z.x = self.x + random.randint(50, 350)
                z.y = self.y + random.randint(-250, 250)

                z.speed = random.uniform(4, 7)

                zombies.append(z)

        # ================= MOVE =================

        self.x -= self.speed

        # ================= DODGE SYSTEM =================

        if self.phase3:

            # né cực mạnh
            self.y += random.randint(-20, 20)

            if random.randint(1, 10) == 1:
                self.dodge_dir *= -1

            self.y += self.dodge_dir * 12

        elif self.rage_mode:

            # né mạnh
            self.y += random.randint(-12, 12)

            if random.randint(1, 20) == 1:
                self.dodge_dir *= -1

            self.y += self.dodge_dir * 5

        else:

            # phase đầu
            self.y += random.randint(-2, 2)

        # ================= LIMIT =================

        if self.y < 180:
            self.y = 180

        if self.y > HEIGHT - 180:
            self.y = HEIGHT - 180

        # ================= GAME OVER =================

        if self.x < -300:

            player_hp = 0

    def draw(self):

        # ================= DRAW BOSS =================

        screen.blit(
            boss_img,
            (self.x - 130, self.y - 170)
        )

        # ================= HP BAR =================

        pygame.draw.rect(
            screen,
            RED,
            (self.x - 150, self.y - 210, 300, 18)
        )

        pygame.draw.rect(
            screen,
            GREEN,
            (
                self.x - 150,
                self.y - 210,
                300 * (self.hp / self.max_hp),
                18
            )
        )

        pygame.draw.rect(
            screen,
            WHITE,
            (self.x - 150, self.y - 210, 300, 18),
            2
        )

        # ================= PHASE TEXT =================

        if self.phase3:

            rage_font = pygame.font.SysFont(
                None,
                55
            )

            rage_text = rage_font.render(
                "PHASE 3",
                True,
                PURPLE
            )

            screen.blit(
                rage_text,
                (self.x - 70, self.y - 260)
            )

        elif self.rage_mode:

            rage_font = pygame.font.SysFont(
                None,
                45
            )

            rage_text = rage_font.render(
                "PHASE 2",
                True,
                RED
            )

            screen.blit(
                rage_text,
                (self.x - 60, self.y - 260)
            )

boss = None

# ================= CREATE FIRST WAVE =================

for _ in range(5):

    zombies.append(Zombie())

# ================= CURSOR =================

hand_x = WIDTH // 2
hand_y = HEIGHT // 2

# ================= MAIN LOOP =================

frame_skip = 0

running = True

while running:

    screen.blit(background, (0, 0))

    # EVENTS

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    # CAMERA

    ret, frame = cap.read()

    if not ret:
        print("Camera failed")
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

    # ================= HAND =================

    if result and result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        action = detect_action(landmarks)

        target_x = int(landmarks[8].x * WIDTH)
        target_y = int(landmarks[8].y * HEIGHT)

        hand_x = int(hand_x * 0.7 + target_x * 0.3)
        hand_y = int(hand_y * 0.7 + target_y * 0.3)

    cursor = (hand_x, hand_y)

    # ================= CHARGE =================

    if action == "CHARGE":

        if not special_ready:

            energy += 1.5

            if energy >= 100:

                energy = 100

                special_ready = True

    # ================= SHOOT =================

    if shoot_cooldown > 0:
        shoot_cooldown -= 1

    # SPECIAL ENERGY SHOT

    if (
        action == "SHOOT"
        and special_ready
        and shoot_cooldown == 0
    ):

        shoot_cooldown = 20

        bullet = EnergyBullet(
            cursor[0],
            cursor[1]
        )

        energy_bullets.append(bullet)

        special_ready = False
        energy = 0

        if gun_sound:
            gun_sound.play()

    # NORMAL SHOOT

    elif (
        action == "SHOOT"
        and shoot_cooldown == 0
    ):

        shoot_cooldown = 10

        pygame.draw.circle(
            screen,
            WHITE,
            cursor,
            35,
            5
        )

        if gun_sound:
            gun_sound.play()

        for z in zombies[:]:

            dx = abs(cursor[0] - z.x)
            dy = abs(cursor[1] - z.y)

            if dx < 60 and dy < 60:

                z.hp -= 50

                for _ in range(15):

                    blood_particles.append(
                        BloodParticle(
                            z.x,
                            z.y
                        )
                    )

                if z.hp <= 0:

                    zombies.remove(z)

                    score += 1

        # HIT BOSS

        if boss:

            dx = abs(cursor[0] - boss.x)
            dy = abs(cursor[1] - boss.y)

            if dx < 180 and dy < 220:

                boss.hp -= 20

    # ================= ENERGY BULLETS =================

    for bullet in energy_bullets[:]:

        bullet.move()

        bullet.draw()

        # phạm vi nổ 500px

        pygame.draw.circle(
            screen,
            BLUE,
            (int(bullet.x), int(bullet.y)),
            500,
            8
        )

        for z in zombies[:]:

            dx = abs(bullet.x - z.x)
            dy = abs(bullet.y - z.y)

            if dx < 500 and dy < 500:

                z.hp -= 500

                for _ in range(40):

                    blood_particles.append(
                        BloodParticle(
                            z.x,
                            z.y
                        )
                    )

                if z.hp <= 0:

                    zombies.remove(z)

                    score += 1

        if boss:

            dx = abs(bullet.x - boss.x)
            dy = abs(bullet.y - boss.y)

            if dx < 500 and dy < 500:

                boss.hp -= 300

                for _ in range(60):

                    blood_particles.append(
                        BloodParticle(
                            boss.x,
                            boss.y
                        )
                    )

                bullet.active = False

        if not bullet.active:

            energy_bullets.remove(bullet)

    # ================= SPAWN BOSS =================

    if len(zombies) == 0 and not boss_spawned:

        boss_spawn_timer += 1

        if boss_spawn_timer >= 50:

            boss_spawned = True

            boss = BossZombie()

            for _ in range(3):

                zombies.append(Zombie())

    # ================= UPDATE =================

    for z in zombies:

        z.move()
        z.draw()

    if boss and boss.hp > 0:

        boss.move()
        boss.draw()

    # ================= WIN =================

    if boss and boss.hp <= 0:

        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))

        screen.blit(overlay, (0, 0))

        win_font = pygame.font.SysFont(None, 120)

        win_text = win_font.render(
            "LEVEL 3 COMPLETE",
            True,
            GREEN
        )

        screen.blit(
            win_text,
            (WIDTH // 2 - 400, HEIGHT // 2)
        )

    # ================= PARTICLES =================

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
        f"KILLS: {score}",
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

    # HP BAR

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

    # ENERGY BAR

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

    # READY TEXT

    if special_ready:

        ready_text = font.render(
            "ENERGY SHOT READY!",
            True,
            BLUE
        )

        screen.blit(ready_text, (20, 220))

    # ================= GAME OVER =================

    if player_hp <= 0:

        over_font = pygame.font.SysFont(
            None,
            100
        )

        game_over_text = over_font.render(
            "GAME OVER",
            True,
            RED
        )

        screen.blit(
            game_over_text,
            (WIDTH // 2 - 250, HEIGHT // 2)
        )

    # ================= CAMERA WINDOW =================

    small_frame = cv2.resize(
        frame,
        (160, 120)
    )

    cv2.putText(
        small_frame,
        action,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "HandCam",
        small_frame
    )

    # ================= UPDATE =================

    pygame.display.update()

    clock.tick(25)

    if cv2.waitKey(1) == 27:
        running = False

# ================= CLEAN =================

cap.release()

cv2.destroyAllWindows()

pygame.quit()