import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up the display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("29 Card Game")

# Load card images
card_images = {}
for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']:
    for card in ['J', '9', 'A', '10', 'K', 'Q', '8', '7']:
        card_name = card + '_' + suit
        card_images[card_name] = pygame.image.load(f"images/{card_name}.png")

# Define card class
class Card:
    def _init_(self, name):
        self.name = name
        self.image = card_images[name]
        self.rect = self.image.get_rect()
        self.power = self.get_power()
        self.points = self.get_points()

    def get_power(self):
        powers = {'J': 7, '9': 6, 'A': 5, '10': 4, 'K': 3, 'Q': 2, '8': 1, '7': 0}
        return powers[self.name.split('_')[0]]

    def get_points(self):
        points = {'J': 3, '9': 2, 'A': 1, '10': 1, 'K': 0, 'Q': 0, '8': 0, '7': 0}
        return points[self.name.split('_')[0]]

# Define player class
class Player:
    def _init_(self, name, hand):
        self.name = name
        self.hand = hand
        self.selected_card = None

    def play_card(self, trump_suit, table_cards):
        if self.selected_card is not None:
            return self.selected_card
        else:
            # Select a card to play
            valid_cards = [card for card in self.hand if card.name.split('_')[1] == trump_suit]
            if not valid_cards:
                return random.choice(self.hand)
            else:
                # Implement a simple strategy for the computer player
                # Play the lowest card of the trump suit if possible
                valid_trump_cards = [card for card in valid_cards if card.power != 6]  # exclude 9
                if valid_trump_cards:
                    return min(valid_trump_cards, key=lambda x: x.power)
                else:
                    return min(self.hand, key=lambda x: x.power)

# Create deck
deck = [Card(name) for name in card_images.keys()]
random.shuffle(deck)

# Deal cards to players
player_hand = [deck.pop() for _ in range(4)]
computer_hand = [deck.pop() for _ in range(4)]

# Create players
player = Player("Player", player_hand)
computer = Player("Computer", computer_hand)

# Define game variables
trump_suit = random.choice(['Hearts', 'Diamonds', 'Clubs', 'Spades'])
trump_card = None
current_winner = None
player_score = 0
computer_score = 0
table_cards = []

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Player selects a card by clicking on it
            mouse_pos = pygame.mouse.get_pos()
            for card in player.hand:
                if card.rect.collidepoint(mouse_pos):
                    player.selected_card = card
                    # Determine winner of the round
                    computer_card = computer.play_card(trump_suit, table_cards)
                    if trump_card:
                        if computer_card.name.split('_')[1] == trump_suit:
                            if player.selected_card.name.split('_')[1] != trump_suit:
                                current_winner = computer
                            else:
                                if card.power > trump_card.power:
                                    current_winner = player
                                else:
                                    current_winner = computer
                        else:
                            if player.selected_card.name.split('_')[1] == trump_suit:
                                current_winner = player
                            else:
                                if card.power > computer_card.power:
                                    current_winner = player
                                else:
                                    current_winner = computer
                    else:
                        if player.selected_card.name.split('_')[1] == trump_suit:
                            current_winner = player
                        else:
                            if card.power > computer_card.power:
                                current_winner = player
                            else:
                                current_winner = computer

                    # Update hands
                    if current_winner == player:
                        player.hand.remove(player.selected_card)
                        player.hand.append(computer_card)
                        player_score += computer_card.points
                    else:
                        computer.hand.remove(computer_card)
                        computer.hand.append(player.selected_card)
                        computer_score += player.selected_card.points

                    # Clear selected card
                    player.selected_card = None

    # Draw the game
    screen.fill(WHITE)
    # Draw player's hand
    for i, card in enumerate(player.hand):
        card.rect.topleft = (20 + i * 120, SCREEN_HEIGHT - 200)
        screen.blit(card.image, card.rect)
    # Draw computer's hand (just the back of the cards)
    for i in range(len(computer.hand)):
        card_back = pygame.image.load("images/card_back.png")
        card_back_rect = card_back.get_rect()
        card_back_rect.topleft = (20 + i * 120, 20)
        screen.blit(card_back, card_back_rect)

    # Display trump suit
    font = pygame.font.SysFont(None, 36)
    text = font.render(f"Trump Suit: {trump_suit}", True, BLACK)
    screen.blit(text, (20, 20))

    # Display scores
    player_text = font.render(f"Player Score: {player_score}", True, BLACK)
    screen.blit(player_text, (20, 50))
    computer_text = font.render(f"Computer Score: {computer_score}", True, BLACK)
    screen.blit(computer_text, (20, 80))

    # Display table cards
    if player.selected_card is not None:
        table_cards.append(player.selected_card)
    if current_winner is not None:
        for i, card in enumerate(table_cards):
            card.rect.topleft = (280 + i * 80, SCREEN_HEIGHT // 2 - 50)
            screen.blit(card.image, card.rect)
        table_cards = []

    pygame.display.flip()

pygame.quit()
sys.exit()