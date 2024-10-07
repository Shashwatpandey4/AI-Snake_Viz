import pygame
import sys
import torch
import numpy as np
from train import SnakeGame, DQN
from scipy.interpolate import interp1d

class Config:
    SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
    PADDING = 20
    TEXT_PADDING = 10
    LABEL_PADDING = 5
    GRID_ROWS = 40
    GRID_COLS = 35
    FPS = 100

class VisualizationHelper:

    @staticmethod
    def draw_colorful_bezier_curve(screen, start_pos, end_pos, start_color, end_color, width=1):
        """
        Args:
            screen (pygame.Surface): The surface to draw on.
            start_pos (tuple): Starting position (x, y).
            end_pos (tuple): Ending position (x, y).
            start_color (tuple): RGB color at the start.
            end_color (tuple): RGB color at the end.
            width (int): Width of the curve.
        """
        control1 = (start_pos[0] + (end_pos[0] - start_pos[0]) // 2, start_pos[1])
        control2 = (start_pos[0] + (end_pos[0] - start_pos[0]) // 2, end_pos[1])

        points = []
        for t in np.linspace(0, 1, 20):
            x = (1 - t)**3 * start_pos[0] + 3 * (1 - t)**2 * t * control1[0] + 3 * (1 - t) * t**2 * control2[0] + t**3 * end_pos[0]
            y = (1 - t)**3 * start_pos[1] + 3 * (1 - t)**2 * t * control1[1] + 3 * (1 - t) * t**2 * control2[1] + t**3 * end_pos[1]
            points.append((int(x), int(y)))

        for i in range(len(points) - 1):
            t = i / (len(points) - 1)
            color = tuple(int(start_color[j] + t * (end_color[j] - start_color[j])) for j in range(3))
            pygame.draw.line(screen, color, points[i], points[i+1], width)

class NetworkVisualizer:

    def __init__(self, model, rect):
        """
        Args:
            model (DQN): The trained model to visualize.
            rect (pygame.Rect): The rectangle to draw the visualization in.
        """
        self.model = model
        self.rect = rect
        self.interp_color = interp1d([0, 1], [(0, 0, 255), (255, 0, 0)], axis=0)

    def draw(self, screen, state):
        """
        Args:
            screen (pygame.Surface): The surface to draw on.
            state (list): The current game state.
        """
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        activations = []
        for layer in self.model.net:
            state_tensor = layer(state_tensor)
            if isinstance(layer, torch.nn.Linear):
                activations.append(state_tensor.detach().numpy())

        layer_count = len(activations)
        layer_positions = np.linspace(self.rect.left + 20, self.rect.right - 20, layer_count)
        max_nodes = max(act.shape[1] for act in activations)
        node_radius = min(5, (self.rect.height - 40) / (2 * max_nodes))

        for i, layer_activations in enumerate(activations):
            layer_size = layer_activations.shape[1]
            node_positions = np.linspace(self.rect.top + 20, self.rect.bottom - 20, layer_size)

            if i < layer_count - 1:
                self._draw_edges(screen, i, layer_activations, activations[i+1], layer_positions, node_positions)

            self._draw_nodes(screen, layer_activations, layer_positions[i], node_positions, node_radius)

    def _draw_edges(self, screen, layer_index, current_layer, next_layer, layer_positions, node_positions):
        """
        Args:
            screen (pygame.Surface): The surface on which to draw the edges.
            layer_index (int): The index of the current layer being drawn.
            current_layer (torch.Tensor): The activations of the current layer, containing the output values for each node.
            next_layer (torch.Tensor): The activations of the next layer, containing the output values for each node.
            layer_positions (numpy.ndarray): The x-coordinates of the center of each layer's visual representation.
            node_positions (numpy.ndarray): The y-coordinates of the center of each node in the current layer.
        """
        next_layer_size = next_layer.shape[1]
        next_node_positions = np.linspace(self.rect.top + 20, self.rect.bottom - 20, next_layer_size)
        for j in range(current_layer.shape[1]):
            for k in range(next_layer_size):
                start_pos = (int(layer_positions[layer_index]), int(node_positions[j]))
                end_pos = (int(layer_positions[layer_index + 1]), int(next_node_positions[k]))

                start_activation = self._normalize_activation(current_layer[0][j], current_layer)
                end_activation = self._normalize_activation(next_layer[0][k], next_layer)

                start_color = tuple(self.interp_color(start_activation).astype(int))
                end_color = tuple(self.interp_color(end_activation).astype(int))

                VisualizationHelper.draw_colorful_bezier_curve(screen, start_pos, end_pos, start_color, end_color)

    def _draw_nodes(self, screen, layer_activations, x_pos, node_positions, node_radius):
        """
        Args:
            screen (pygame.Surface): The surface on which to draw the nodes.
            layer_activations (torch.Tensor): The activations of the layer, containing the output values for each node.
            x_pos (int): The x-coordinate for the center of the layer's visual representation.
            node_positions (numpy.ndarray): The y-coordinates for the positions of each node within the layer.
            node_radius (int): The radius of the circles representing the nodes.
        """
        for j, activation in enumerate(layer_activations[0]):
            activation_value = self._normalize_activation(activation, layer_activations)
            color = tuple(self.interp_color(activation_value).astype(int))
            node_pos = (int(x_pos), int(node_positions[j]))
            pygame.draw.circle(screen, color, node_pos, node_radius)
            pygame.draw.circle(screen, 0xFFFFFF, node_pos, node_radius, 1)

    @staticmethod
    def _normalize_activation(activation, layer_activations):
        return (activation - np.min(layer_activations)) / (np.max(layer_activations) - np.min(layer_activations))

class GameVisualizer:

    def __init__(self, game, rect):
        """
        Initialize the GameVisualizer.

        Args:
            game (SnakeGame): The game instance to visualize.
            rect (pygame.Rect): The rectangle to draw the game in.
        """
        self.game = game
        self.rect = rect
        self.box_width = rect.width // Config.GRID_COLS
        self.box_height = rect.height // Config.GRID_ROWS

    def draw(self, screen):
        """
        Args:
            screen (pygame.Surface): The surface to draw on.
        """
        self._draw_grid(screen)
        self._draw_snake(screen)
        self._draw_food(screen)

    def _draw_grid(self, screen):
        for row in range(Config.GRID_ROWS):
            for col in range(Config.GRID_COLS):
                box_x = self.rect.left + col * self.box_width
                box_y = self.rect.top + row * self.box_height
                pygame.draw.rect(screen, 0x787878, (box_x, box_y, self.box_width, self.box_height), 1)

    def _draw_snake(self, screen):
        for segment in self.game.snake:
            pygame.draw.rect(screen, 0x00FF00, (
                self.rect.left + segment[1] * self.box_width, 
                self.rect.top + segment[0] * self.box_height, 
                self.box_width, self.box_height
            ))

    def _draw_food(self, screen):
        pygame.draw.rect(screen, 0xFF0000, (
            self.rect.left + self.game.food[1] * self.box_width, 
            self.rect.top + self.game.food[0] * self.box_height, 
            self.box_width, self.box_height
        ))

class SnakeAIVisualization:

    def __init__(self):
        """Initialize the SnakeAIVisualization."""
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption('Snake AI Visualization')
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 36)  

        self.model = self._load_model()
        self.game = SnakeGame(width=Config.GRID_COLS, height=Config.GRID_ROWS)

        left_rect = pygame.Rect(Config.PADDING, Config.PADDING, 
                                Config.SCREEN_WIDTH // 2 - Config.PADDING, 
                                Config.SCREEN_HEIGHT - Config.PADDING * 2 - 50)
        right_rect = pygame.Rect(Config.SCREEN_WIDTH // 2 + Config.PADDING, Config.PADDING, 
                                 Config.SCREEN_WIDTH // 2 - Config.PADDING * 2, 
                                 Config.SCREEN_HEIGHT - Config.PADDING * 2 - 50)

        self.game_visualizer = GameVisualizer(self.game, left_rect)
        self.network_visualizer = NetworkVisualizer(self.model, right_rect)

    @staticmethod
    def _load_model():
        model = DQN(11, 32, 3)
        model.load_state_dict(torch.load('snake_ai_model.pth'))
        model.eval()
        return model

    def get_action(self, state):
        """
        Get the next action from the model.

        Args:
            state (list): The current game state.

        Returns:
            int: The action to take.
        """
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(state)
        return torch.argmax(prediction).item()

    def run(self):
        """Run the main game loop."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            state = self.game._get_state()
            action = self.get_action(state)
            _, _, done = self.game.step(action)

            self.screen.fill(0x000000)
            self.game_visualizer.draw(self.screen)
            self.network_visualizer.draw(self.screen, state)

            score_text = self.font.render(f'Score: {self.game.score}', True, (255, 255, 255), (0, 0, 0))  
            self.screen.blit(score_text, (Config.PADDING, Config.SCREEN_HEIGHT - Config.PADDING - 36))

            pygame.display.flip()
            self.clock.tick(Config.FPS)

            if done:
                self.game.reset()

if __name__ == "__main__":
    visualization = SnakeAIVisualization()
    visualization.run()