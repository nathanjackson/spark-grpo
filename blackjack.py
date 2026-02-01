import random

class Blackjack:
    def __init__(self):
        self.deck = self._create_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.game_over = False

    def _create_deck(self):
        """Create and shuffle a standard 52-card deck."""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        deck = [(rank, suit) for rank in ranks for suit in suits]
        random.shuffle(deck)
        return deck

    def _draw_card(self):
        """Draw one card from the deck."""
        return self.deck.pop()

    def _hand_value(self, hand):
        """Compute hand value with proper Ace handling."""
        value = 0
        aces = 0

        for rank, _ in hand:
            if rank in ['J', 'Q', 'K']:
                value += 10
            elif rank == 'A':
                value += 11
                aces += 1
            else:
                value += int(rank)

        # Convert Aces from 11 → 1 if needed
        while value > 21 and aces:
            value -= 10
            aces -= 1

        return value

    def start_round(self):
        """Start a new round: deal two cards to player and dealer."""
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        self.game_over = False

    def hit(self):
        """Give the player a card and auto-check for bust."""
        if self.game_over:
            return "Game is already over."

        self.player_hand.append(self._draw_card())
        if self._hand_value(self.player_hand) > 21:
            self.game_over = True
            return "Player busts!"
        return "Player hits."

    def stand(self):
        """Player ends their turn; dealer draws until 17+."""
        if self.game_over:
            return "Game is already over."

        while self._hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self._draw_card())

        self.game_over = True
        return self._determine_winner()

    def _determine_winner(self):
        p_val = self._hand_value(self.player_hand)
        d_val = self._hand_value(self.dealer_hand)

        if d_val > 21:
            return "Dealer busts! Player wins."
        if p_val > d_val:
            return "Player wins."
        if p_val < d_val:
            return "Dealer wins."
        return "Push (tie)."

    def get_state(self):
        """Return current game state for UI/logic use."""
        return {
            "player_hand": self.player_hand,
            "dealer_hand": self.dealer_hand,
            "player_value": self._hand_value(self.player_hand),
            "dealer_value": self._hand_value(self.dealer_hand) if self.game_over else "Hidden",
            "game_over": self.game_over
        }

