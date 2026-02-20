import random

import torch

import torch.nn.functional as F

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

def _build_action_token_sequences(tokenizer):
    # Include common variants with/without leading space.
    variants = ["HIT", " HIT", "STAY", " STAY"]
    sequences = []
    for text in variants:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            sequences.append(ids)
    # Deduplicate while preserving order.
    deduped = []
    for seq in sequences:
        if seq not in deduped:
            deduped.append(seq)
    return deduped


def _make_prefix_allowed_tokens_fn(prompt_len, action_seqs, eos_token_id):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        # Only constrain the generated portion after the prompt.
        gen = input_ids[prompt_len:].tolist()
        # If nothing generated yet, allow the first token of any action.
        if len(gen) == 0:
            return sorted({seq[0] for seq in action_seqs if len(seq) > 0})
        allowed = set()
        for seq in action_seqs:
            if gen == seq:
                # Allow EOS (or any token that ends generation) once full action is matched.
                if eos_token_id is not None:
                    allowed.add(eos_token_id)
                continue
            if len(gen) < len(seq) and gen == seq[:len(gen)]:
                allowed.add(seq[len(gen)])
        return sorted(allowed) if allowed else ([] if eos_token_id is None else [eos_token_id])
    return prefix_allowed_tokens_fn

def generate_trajectory(game, tokenizer, model, temperature: float = 1.0):
    device = model.device
    messages = [
        { "role": "system", "content": ("You are playing blackjack. At each "
            "step you'll be given the state of the game. Respond with HIT or "
            "STAY. Do not add commentary.")
        }
    ]
    action_mask = None
    token_rewards = None

    invalid_action = False
    sequence_ids = None
    #print("start game")
    while not game.get_state()["game_over"]:
        visible_state = {
            "your_hand": game.get_state()["player_hand"],
            "dealer_hand": ["Hidden"] + game.get_state()["dealer_hand"][1:]
        }
        state_str = str(visible_state)
        #print(state_str)
        messages.append({ "role": "user", "content": state_str })

        messages_text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True)
        #print(messages_text)
        messages_encoding = tokenizer(messages_text, return_tensors="pt").to(device)

        prompt_len = messages_encoding.input_ids.shape[1]
        action_seqs = _build_action_token_sequences(tokenizer)
        prefix_allowed_tokens_fn = _make_prefix_allowed_tokens_fn(
            prompt_len,
            action_seqs,
            tokenizer.eos_token_id,
        )
        with torch.no_grad():
            outputs = model.generate(
                **messages_encoding,
                max_new_tokens=4,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                return_dict_in_generate=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        sequence_ids = outputs.sequences[0]
        action_ids = outputs.sequences[0, prompt_len:].unsqueeze(0)
        if action_mask is None:
            action_mask = torch.zeros(outputs.sequences.shape[1], dtype=torch.bool).to(device)
        else:
            new_ids = outputs.sequences[0, action_mask.shape[0]:]
            action_mask = torch.cat((action_mask, torch.zeros(new_ids.shape[0], dtype=torch.bool).to(device)), dim=0)
        response_ids = outputs.sequences[0, prompt_len:]
        if token_rewards is None:
            token_rewards = torch.zeros(outputs.sequences.shape[1], dtype=torch.float32, device=device)
        else:
            new_ids = outputs.sequences[0, token_rewards.shape[0]:]
            token_rewards = torch.cat((token_rewards, torch.zeros(new_ids.shape[0], dtype=torch.float32, device=device)), dim=0)
        #action_mask[messages_encoding.input_ids.shape[1]:][response_ids != tokenizer.eos_token_id] = 1
        #print(action_mask)
        action = tokenizer.batch_decode(action_ids, skip_special_tokens=True)[0]
        action = action.strip().upper()
        #print(f"Action: {action}")
        #print(f"Action token IDs: {action_ids[0].tolist()}")
        #print(f"Mask for these tokens: {action_mask[messages_encoding.input_ids.shape[1]:messages_encoding.input_ids.shape[1]+len(action_ids[0])].tolist()}")
        messages.append({ "role": "assistant", "content": action})
        #print(action)

        if action == "HIT":
            #print("model hits")
            observation = game.hit()
            action_mask[messages_encoding.input_ids.shape[1]:][response_ids != tokenizer.eos_token_id] = 1
        elif action == "STAY":
            #print("model stays")
            observation = game.stand()
            action_mask[messages_encoding.input_ids.shape[1]:][response_ids != tokenizer.eos_token_id] = 1
        else:
            #print("invalid action")
            invalid_action = True
            action_mask[messages_encoding.input_ids.shape[1]:][response_ids != tokenizer.eos_token_id] = 1
            #action_mask[messages_encoding.input_ids.shape[1]:] = 1
            break

    if invalid_action:
        # agent did not play properly
        #print("game did not finish")
        reward = -2.
    elif "push" in observation.lower():
        #print("push")
        reward = 1. 
    elif "player wins" in observation.lower():
        #print("agent win")
        reward = 2.
    else:
        #print("agent loss")
        reward = -1.

    if token_rewards is None:
        token_rewards = torch.zeros(sequence_ids.shape[0], dtype=torch.float32, device=device)
    token_rewards[action_mask] += reward

    #print("Reward:", reward)
    #print("Mask:", action_mask)

    with torch.no_grad():
        logits = model(sequence_ids.unsqueeze(0)).logits
        logprobs = F.log_softmax(logits, dim=-1)
        old_logprobs = logprobs[:, :-1, :].gather(
            -1, sequence_ids.unsqueeze(0)[:, 1:].unsqueeze(-1)
        ).squeeze(-1).squeeze(0)

    return {
        "messages": messages,
        "sequence_ids": sequence_ids,
        "action_mask": action_mask,
        "old_logprobs": old_logprobs,
        "token_rewards": token_rewards,
        "reward": reward
    }

