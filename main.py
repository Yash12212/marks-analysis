import pygame
import random
import pickle
import os

pygame.init()

# --- UI Constants and Colors ---
FONT_NAME = 'Arial'
TITLE_FONT_SIZE = 40           # Smaller title font
BUTTON_FONT_SIZE = 24          # Smaller button font
TEXT_COLOR = (240, 240, 240)     # Brighter text
BACKGROUND_TOP_COLOR = (40, 40, 40)
BACKGROUND_BOTTOM_COLOR = (10, 10, 10)
BUTTON_COLOR_NORMAL = (50, 150, 50)
BUTTON_COLOR_HOVER = (70, 170, 70)
BUTTON_SHADOW_COLOR = (0, 0, 0)
SLIDER_BG_COLOR = (100, 100, 100)
SLIDER_HANDLE_COLOR = (150, 150, 250)
PADDLE_COLOR = (200, 200, 200)
BALL_COLOR = (255, 100, 100)
SCORE_COLOR = TEXT_COLOR

# --- Layout Constants ---
MARGIN_TOP = 30
MARGIN_BOTTOM = 30
MARGIN_LEFT = 30
MARGIN_RIGHT = 30
MENU_TITLE_Y = 80
MENU_GAP_AFTER_TITLE = 10       # Reduced gap below title
INSTRUCTION_GAP = 5             # Reduced gap for instruction text
MENU_BUTTON_SPACING = 10        # Reduced vertical space between buttons
BUTTON_WIDTH = 240
BUTTON_HEIGHT = 50              # Reduced button height

# Screen dimensions and setup
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Genetic Pong")
clock = pygame.time.Clock()

# Game constants
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
BALL_SIZE = 10

# File to save/load the population
POPULATION_FILE = "population.pkl"

# ---------------------------------
# Utility: Draw a vertical gradient background
# ---------------------------------
def draw_gradient_background(surface, top_color, bottom_color):
    height = surface.get_height()
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (0, y), (surface.get_width(), y))

# ----------------------------
# Genetic Agent & Game Classes
# ----------------------------
class Agent:
    def __init__(self, weight=None, bias=None):
        self.weight = weight if weight is not None else random.uniform(-1, 1)
        self.bias = bias if bias is not None else random.uniform(-10, 10)
        self.fitness = 0

    def decide(self, paddle_center, ball_y):
        decision = self.weight * (ball_y - paddle_center) + self.bias
        return 1 if decision > 0 else -1

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.vx = random.choice([-4, 4])
        self.vy = random.choice([-3, -2, -1, 1, 2, 3])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        if self.y <= 0 or self.y >= HEIGHT - BALL_SIZE:
            self.vy = -self.vy

# Genetic algorithm operators
def crossover(parent1, parent2):
    child_weight = (parent1.weight + parent2.weight) / 2
    child_bias = (parent1.bias + parent2.bias) / 2
    return Agent(child_weight, child_bias)

def mutate(agent, mutation_rate=0.1):
    if random.random() < mutation_rate:
        agent.weight += random.uniform(-0.5, 0.5)
    if random.random() < mutation_rate:
        agent.bias += random.uniform(-5, 5)

def evolve(population):
    population.sort(key=lambda a: a.fitness, reverse=True)
    new_population = population[:2]
    while len(new_population) < len(population):
        parent1 = random.choice(population[:min(10, len(population))])
        parent2 = random.choice(population[:min(10, len(population))])
        child = crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
    return new_population

# -----------------------------------
# Updated UI Classes: Slider and Menu Button
# -----------------------------------
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, start_val):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = start_val
        self.handle_width = 20  # Increased handle width for better usability
        self.handle_rect = pygame.Rect(x, y, self.handle_width, h)
        self.dragging = False
        self.update_handle_position()

    def update_handle_position(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect.x = self.rect.x + int(ratio * (self.rect.w - self.handle_width))

    def draw(self, surface):
        pygame.draw.rect(surface, SLIDER_BG_COLOR, self.rect, border_radius=5)
        pygame.draw.rect(surface, SLIDER_HANDLE_COLOR, self.handle_rect, border_radius=5)
        tooltip_font = pygame.font.SysFont(FONT_NAME, 18)
        tooltip_text = tooltip_font.render(f"{self.value:.1f}x", True, TEXT_COLOR)
        tooltip_rect = tooltip_text.get_rect(midtop=(self.handle_rect.centerx, self.handle_rect.bottom + 4))

        surface.blit(tooltip_text, tooltip_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
            elif self.rect.collidepoint(event.pos):
                self.handle_rect.x = event.pos[0] - self.handle_width // 2
                self.update_value_from_handle()
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.handle_rect.x = max(self.rect.x, min(event.pos[0] - self.handle_width // 2, self.rect.x + self.rect.w - self.handle_width))
            self.update_value_from_handle()

    def update_value_from_handle(self):
        ratio = (self.handle_rect.x - self.rect.x) / (self.rect.w - self.handle_width)
        self.value = self.min_val + ratio * (self.max_val - self.min_val)

    def get_value(self):
        return self.value

class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = pygame.font.SysFont(FONT_NAME, BUTTON_FONT_SIZE)
        self.normal_color = BUTTON_COLOR_NORMAL
        self.hover_color = BUTTON_COLOR_HOVER

    def draw(self, surface, hovered):
        scale = 1.05 if hovered else 1.0
        scaled_rect = self.rect.inflate(self.rect.width * (scale - 1), self.rect.height * (scale - 1))
        shadow_rect = scaled_rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        pygame.draw.rect(surface, BUTTON_SHADOW_COLOR, shadow_rect, border_radius=8)
        color = self.hover_color if hovered else self.normal_color
        pygame.draw.rect(surface, color, scaled_rect, border_radius=8)
        text_surface = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=scaled_rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

    def is_hovered(self, pos):
        return self.rect.collidepoint(pos)

# -----------------------------------
# Simulation, Training, and Testing
# -----------------------------------
def simulate_agent(agent, speed_multiplier=1, render=True):
    paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
    ball = Ball()
    hits = 0
    sim_steps = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        move = agent.decide(paddle_y + PADDLE_HEIGHT / 2, ball.y)
        paddle_y += move * 4
        paddle_y = max(0, min(paddle_y, HEIGHT - PADDLE_HEIGHT))
        ball.update()

        if ball.x <= PADDLE_WIDTH:
            if paddle_y < ball.y < paddle_y + PADDLE_HEIGHT:
                ball.vx = -ball.vx
                hits += 1
            else:
                running = False
        if ball.x >= WIDTH - BALL_SIZE:
            ball.vx = -ball.vx

        if render:
            draw_gradient_background(screen, BACKGROUND_TOP_COLOR, BACKGROUND_BOTTOM_COLOR)
            pygame.draw.rect(screen, PADDLE_COLOR, (0, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
            pygame.draw.rect(screen, BALL_COLOR, (ball.x, ball.y, BALL_SIZE, BALL_SIZE))
            font = pygame.font.SysFont(FONT_NAME, 24)
            score_text = font.render(f'Hits: {hits}', True, SCORE_COLOR)
            screen.blit(score_text, (WIDTH // 2 - 50, 20))
            pygame.display.flip()
        sim_steps += 1
        if sim_steps > 1000:
            running = False
        clock.tick(int(60 * speed_multiplier))
    agent.fitness = hits
    return hits

def training_loop(population, slider):
    generations = 20
    font = pygame.font.SysFont(FONT_NAME, 24)
    for gen in range(generations):
        print(f"--- Generation {gen} ---")
        for i, agent in enumerate(population):
            print(f"Evaluating Agent {i}...", end=" ")
            paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
            ball = Ball()
            hits = 0
            sim_steps = 0
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    slider.handle_event(event)
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return population

                move = agent.decide(paddle_y + PADDLE_HEIGHT / 2, ball.y)
                paddle_y += move * 4
                paddle_y = max(0, min(paddle_y, HEIGHT - PADDLE_HEIGHT))
                ball.update()
                if ball.x <= PADDLE_WIDTH:
                    if paddle_y < ball.y < paddle_y + PADDLE_HEIGHT:
                        ball.vx = -ball.vx
                        hits += 1
                    else:
                        running = False
                if ball.x >= WIDTH - BALL_SIZE:
                    ball.vx = -ball.vx

                draw_gradient_background(screen, BACKGROUND_TOP_COLOR, BACKGROUND_BOTTOM_COLOR)
                pygame.draw.rect(screen, PADDLE_COLOR, (0, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
                pygame.draw.rect(screen, BALL_COLOR, (ball.x, ball.y, BALL_SIZE, BALL_SIZE))
                score_text = font.render(f'Hits: {hits}', True, SCORE_COLOR)
                screen.blit(score_text, (WIDTH // 2 - 50, 20))
                slider.draw(screen)
                slider_label = font.render("Speed", True, SCORE_COLOR)
                screen.blit(slider_label, (slider.rect.x, slider.rect.y - 25))
                pygame.display.flip()
                sim_steps += 1
                if sim_steps > 1000:
                    running = False
                clock.tick(int(60 * slider.get_value()))
            agent.fitness = hits
            print(f"Hits: {hits}")
            pygame.time.wait(300)
        best_agent = max(population, key=lambda a: a.fitness)
        print(f"Best Hits in Generation {gen}: {best_agent.fitness}")
        pygame.time.wait(1000)
        population = evolve(population)
        with open(POPULATION_FILE, "wb") as f:
            pickle.dump(population, f)
    return population

def testing_loop(agent, slider):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            slider.handle_event(event)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        simulate_agent(agent, speed_multiplier=slider.get_value(), render=True)

# ---------------------
# Main Menu and Control (Further Improved Layout)
# ---------------------
def menu_loop():
    title_font = pygame.font.SysFont(FONT_NAME, TITLE_FONT_SIZE)
    title_text = title_font.render("Genetic Pong", True, TEXT_COLOR)
    title_rect = title_text.get_rect(center=(WIDTH // 2, MENU_TITLE_Y))
    
    instruction_font = pygame.font.SysFont(FONT_NAME, 16)
    instruction_text = instruction_font.render("Train, load, or test your evolved pong agent", True, TEXT_COLOR)
    instruction_rect = instruction_text.get_rect(center=(WIDTH // 2, title_rect.bottom + MENU_GAP_AFTER_TITLE + INSTRUCTION_GAP + 10))
    
    button_container_top = instruction_rect.bottom + MENU_GAP_AFTER_TITLE
    buttons = []
    button_texts = ["New Training", "Load Training", "Test Best Agent", "Quit"]
    for i, text in enumerate(button_texts):
        x = WIDTH // 2 - BUTTON_WIDTH // 2
        y = button_container_top + i * (BUTTON_HEIGHT + MENU_BUTTON_SPACING)
        buttons.append(Button((x, y, BUTTON_WIDTH, BUTTON_HEIGHT), text))
    
    slider_width = 200
    slider_height = 20
    slider_x = WIDTH // 2 - slider_width // 2
    slider_y = HEIGHT - MARGIN_BOTTOM - slider_height - 10
    slider = Slider(slider_x, slider_y, slider_width, slider_height, 1, 5, 1)
    
    menu_message = ""
    message_timer = 0

    while True:
        draw_gradient_background(screen, BACKGROUND_TOP_COLOR, BACKGROUND_BOTTOM_COLOR)
        screen.blit(title_text, title_rect)
        screen.blit(instruction_text, instruction_rect)
        
        mouse_pos = pygame.mouse.get_pos()
        hovered_button = None
        for btn in buttons:
            if btn.is_hovered(mouse_pos):
                hovered_button = btn
                break

        for btn in buttons:
            btn.draw(screen, hovered_button == btn)
        
        slider.draw(screen)
        slider_label = pygame.font.SysFont(FONT_NAME, 24).render("Speed", True, TEXT_COLOR)
        screen.blit(slider_label, (slider.rect.x, slider.rect.y - 25))
        
        if menu_message and pygame.time.get_ticks() < message_timer + 2000:
            message_surface = pygame.font.SysFont(FONT_NAME, 24).render(menu_message, True, (255, 100, 100))
            message_rect = message_surface.get_rect(center=(WIDTH // 2, slider.rect.y - 40))
            screen.blit(message_surface, message_rect)
        elif menu_message:
            menu_message = ""
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            slider.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if buttons[0].is_clicked(pos):
                    population = [Agent() for _ in range(20)]
                    return "training", population, slider
                elif buttons[1].is_clicked(pos):
                    if os.path.exists(POPULATION_FILE):
                        with open(POPULATION_FILE, "rb") as f:
                            population = pickle.load(f)
                        return "training", population, slider
                    else:
                        menu_message = "No saved training data found!"
                        message_timer = pygame.time.get_ticks()
                elif buttons[2].is_clicked(pos):
                    if os.path.exists(POPULATION_FILE):
                        with open(POPULATION_FILE, "rb") as f:
                            population = pickle.load(f)
                        best_agent = max(population, key=lambda a: a.fitness)
                        return "testing", best_agent, slider
                    else:
                        menu_message = "No saved agent to test!"
                        message_timer = pygame.time.get_ticks()
                elif buttons[3].is_clicked(pos):
                    pygame.quit()
                    exit()
        
        pygame.display.flip()
        clock.tick(60)

def main():
    state = "menu"
    population = None
    best_agent = None
    slider = None

    while True:
        if state == "menu":
            action, data, slider = menu_loop()
            if action == "training":
                population = data
                state = "training"
            elif action == "testing":
                best_agent = data
                state = "testing"
        elif state == "training":
            population = training_loop(population, slider)
            state = "menu"
        elif state == "testing":
            testing_loop(best_agent, slider)
            state = "menu"

if __name__ == '__main__':
    main()
