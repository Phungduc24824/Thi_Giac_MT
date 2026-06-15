from ast import main
import os
import pygame
import sys
pygame.init()

WIDTH = 1550 
HEIGHT = 800

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Zombie Hand Shooter")

clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (60, 60, 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
background_path = os.path.join(assets_dir, "bg_city.jpg")

if not os.path.exists(background_path):
    for fallback_name in ["city_1.jpg", "city_2.jpg", "background.png"]:
        candidate = os.path.join(assets_dir, fallback_name)
        if os.path.exists(candidate):
            background_path = candidate
            break
    else:
        raise FileNotFoundError(
            f"No background image found in {assets_dir}."
            " Expected one of: bg_city.jpg, city_1.jpg, city_2.jpg, background.png"
        )

background = pygame.image.load(background_path).convert()

background = pygame.transform.scale(
    background,
    (WIDTH, HEIGHT)
)

title_font = pygame.font.SysFont("Arial", 70)
menu_font = pygame.font.SysFont("Arial", 40)
story_font = pygame.font.SysFont("Arial", 28)
start_font = pygame.font.SysFont("Arial", 55)

story_mode = True

story_lines = [
    "NAM 2045...",
    "",
    "Mot loai virus bi ro ri tu phong thi nghiem sinh hoc.",
    "",
    "Chi trong 7 ngay...",
    "Ca thanh pho da bi bien thanh dia nguc.",
    "",
    "Con nguoi bi bien doi thanh ZOMBIE khat mau.",
    "",
    "Ban la nguoi song sot cuoi cung.",
    "",
    "Hay dung ky nang tay cua ban",
    "de tieu diet toan bo zombie.",
]

selected_level = None

level1_rect = pygame.Rect(100, 220, 250, 100)
level2_rect = pygame.Rect(100, 380, 250, 100)
level3_rect = pygame.Rect(100, 540, 250, 100)

start_button = pygame.Rect(850, 520, 300, 120)

story_next_button = pygame.Rect(900, 560, 280, 90)

running = True

while running:

    screen.blit(background, (0, 0))

    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(140)
    overlay.fill(BLACK)

    screen.blit(overlay, (0, 0))

    # =====================================================
    # STORY SCREEN
    # =====================================================

    if story_mode:

        title = title_font.render(
            "THE ZOMBIE APOCALYPSE",
            True,
            RED
        )

        screen.blit(title, (220, 60))

        y = 180

        for line in story_lines:

            text = story_font.render(
                line,
                True,
                WHITE
            )

            screen.blit(text, (170, y))

            y += 40

        # ================= BUTTON PLAY =================

        mouse_pos = pygame.mouse.get_pos()

        button_color = GREEN

        if story_next_button.collidepoint(mouse_pos):
            button_color = (0, 200, 0)

        pygame.draw.rect(
            screen,
            button_color,
            story_next_button,
            border_radius=20
        )

        pygame.draw.rect(
            screen,
            WHITE,
            story_next_button,
            4,
            border_radius=20
        )

        play_text = start_font.render(
            "PLAY >>",
            True,
            WHITE
        )

        screen.blit(play_text, (950, 575))

    # =====================================================
    # MENU SCREEN
    # =====================================================

    else:

        title = title_font.render(
            "ZOMBIE HAND SHOOTER",
            True,
            RED
        )

        screen.blit(title, (210, 60))

        level1_color = GRAY
        level2_color = GRAY
        level3_color = GRAY

        if selected_level == 1:
            level1_color = RED

        if selected_level == 2:
            level2_color = RED

        if selected_level == 3:
            level3_color = RED

        # ================= LEVEL 1 =================

        pygame.draw.rect(
            screen,
            level1_color,
            level1_rect,
            border_radius=20
        )

        pygame.draw.rect(
            screen,
            WHITE,
            level1_rect,
            3,
            border_radius=20
        )

        text1 = menu_font.render(
            "LEVEL 1",
            True,
            WHITE
        )

        screen.blit(text1, (150, 250))

        # ================= LEVEL 2 =================

        pygame.draw.rect(
            screen,
            level2_color,
            level2_rect,
            border_radius=20
        )

        pygame.draw.rect(
            screen,
            WHITE,
            level2_rect,
            3,
            border_radius=20
        )

        text2 = menu_font.render(
            "LEVEL 2",
            True,
            WHITE
        )

        screen.blit(text2, (150, 410))

        # ================= LEVEL 3 =================

        pygame.draw.rect(
            screen,
            level3_color,
            level3_rect,
            border_radius=20
        )

        pygame.draw.rect(
            screen,
            WHITE,
            level3_rect,
            3,
            border_radius=20
        )

        text3 = menu_font.render(
            "LEVEL 3",
            True,
            WHITE
        )

        screen.blit(text3, (150, 570))

        # ================= INFO =================

        info = story_font.render(
            "CHON MAN CHOI",
            True,
            YELLOW
        )

        screen.blit(info, (820, 180))

        if selected_level:

            selected_text = story_font.render(
                f"DA CHON LEVEL {selected_level}",
                True,
                WHITE
            )

            screen.blit(selected_text, (820, 240))

            # ================= START BUTTON =================

            mouse_pos = pygame.mouse.get_pos()

            start_color = GREEN

            if start_button.collidepoint(mouse_pos):
                start_color = (0, 200, 0)

            pygame.draw.rect(
                screen,
                start_color,
                start_button,
                border_radius=25
            )

            pygame.draw.rect(
                screen,
                WHITE,
                start_button,
                4,
                border_radius=25
            )

            start_text = start_font.render(
                "START",
                True,
                WHITE
            )

            screen.blit(start_text, (930, 550))

    # =====================================================
    # EVENTS
    # =====================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        # ================= CLICK =================

        if event.type == pygame.MOUSEBUTTONDOWN:

            mouse_pos = pygame.mouse.get_pos()

            # ================= STORY -> MENU =================

            if story_mode:

                if story_next_button.collidepoint(mouse_pos):
                    story_mode = False

            # ================= MENU =================

            else:

                if level1_rect.collidepoint(mouse_pos):
                    selected_level = 1

                if level2_rect.collidepoint(mouse_pos):
                    selected_level = 2

                if level3_rect.collidepoint(mouse_pos):
                    selected_level = 3

                # ================= START GAME =================

                if selected_level:

                    if start_button.collidepoint(mouse_pos):

                        print(f"START LEVEL {selected_level}")

                        if selected_level == 1:
                            import level1
                            import level2
                            
                            # Vòng lặp điều khiển trạng thái sau khi chơi xong Level 1
                            current_action = "play"
                            while current_action == "play" or current_action == "replay":
                                # Chạy Level 1 và hứng kết quả trả về từ các nút bấm
                                result = level1.run_game() 
                                
                                if result == "replay":
                                    current_action = "replay" # Tiếp tục vòng lặp để chơi lại Lvl 1
                                elif result == "next_level":
                                    current_action = "next"
                                    # Chạy thẳng sang Level 2
                                    level2.run_game() 
                                else:
                                    break # Thoát vòng lặp quay về menu chính nếu tắt ngang

                        elif selected_level == 2:

                            import level2
                            import level3
                            
                            # Vòng lặp điều khiển trạng thái sau khi chơi xong Level 1
                            current_action = "play"
                            while current_action == "play" or current_action == "replay":
                                # Chạy Level 1 và hứng kết quả trả về từ các nút bấm
                                result = level2.run_game() 
                                
                                if result == "replay":
                                    current_action = "replay" # Tiếp tục vòng lặp để chơi lại Lvl 1
                                elif result == "next_level":
                                    current_action = "next"
                                    # Chạy thẳng sang Level 2
                                    level3.run_game() 
                                else:
                                    break
                        elif selected_level == 3:
                            import level3
                            
                            # Vòng lặp điều khiển trạng thái sau khi chơi xong Level 3
                            current_action = "play"
                            while current_action == "play" or current_action == "replay":
                                # Chạy Level 3 và hứng kết quả trả về từ các nút bấm
                                result = level3.run_game() 
                                
                                if result == "replay":
                                    current_action = "replay" # Tiếp tục vòng lặp để chơi lại Lvl 3
                                elif result == "next_level":
                                    current_action = "next"
                                    # Chạy thẳng sang Level 3
                                    main.run_game() 
                                else:
                                    break

    pygame.display.flip()

    clock.tick(25)

pygame.quit()
sys.exit() 