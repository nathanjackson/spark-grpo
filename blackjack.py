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


def _build_action_candidates(tokenizer):
    action_variants = {
        "HIT": ["HIT", " HIT"],
        "STAY": ["STAY", " STAY"],
    }
    candidates = []
    for action, variants in action_variants.items():
        token_variants = []
        for text in variants:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids and ids not in token_variants:
                token_variants.append(ids)
        if token_variants:
            candidates.append({
                "action": action,
                "variants": token_variants,
            })
    return candidates

def _score_token_sequence(model, prompt_ids, attention_mask, token_ids):
    input_ids = prompt_ids
    attn_mask = attention_mask
    total_logprob = torch.tensor(0.0, device=prompt_ids.device)

    for token_id in token_ids:
        logits = model(input_ids, attention_mask=attn_mask).logits[:, -1, :]
        logprobs = F.log_softmax(logits, dim=-1)
        total_logprob = total_logprob + logprobs[0, token_id]

        next_token = torch.tensor([[token_id]], device=prompt_ids.device, dtype=prompt_ids.dtype)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        attn_mask = torch.cat((attn_mask, torch.ones_like(next_token)), dim=1)

    return total_logprob


def _score_action_candidates(messages_encoding, tokenizer, model):
    candidates = _build_action_candidates(tokenizer)
    if not candidates:
        raise ValueError("Tokenizer could not encode any valid blackjack actions.")

    action_scores = []
    with torch.no_grad():
        for candidate in candidates:
            variant_scores = []
            for token_ids in candidate["variants"]:
                score = _score_token_sequence(
                    model,
                    messages_encoding.input_ids,
                    messages_encoding.attention_mask,
                    token_ids,
                )
                variant_scores.append(score)

            stacked_scores = torch.stack(variant_scores)
            action_scores.append(torch.logsumexp(stacked_scores, dim=0))

    return candidates, torch.stack(action_scores)


def _action_instructions():
    return (
        "You are playing blackjack. At each step you'll be given the state "
        "of the game. Respond with HIT or STAY. Do not add commentary."
    )


def _visible_state(game):
    state = game.get_state()
    return {
        "your_hand": state["player_hand"],
        "dealer_hand": ["Hidden"] + state["dealer_hand"][1:],
    }


def _user_turn_content(game, include_instructions):
    state_text = str(_visible_state(game))
    if include_instructions:
        # Mistral's chat template folds or drops `system` content once assistant
        # turns are present, which breaks prompt/action span reconstruction.
        return f"{_action_instructions()}\n\n{state_text}"
    return state_text


def _build_training_sequence(tokenizer, messages, device):
    final_text = tokenizer.apply_chat_template(messages, tokenize=False)
    final_encoding = tokenizer(
        final_text,
        return_tensors="pt",
        return_special_tokens_mask=True,
    ).to(device)
    sequence_ids = final_encoding.input_ids[0]
    action_mask = torch.zeros(sequence_ids.shape[0], dtype=torch.bool, device=device)

    replay_messages = []
    msg_idx = 0
    while msg_idx + 1 < len(messages):
        replay_messages.append(messages[msg_idx])
        prompt_text = tokenizer.apply_chat_template(
            replay_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_encoding = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = prompt_encoding.input_ids.shape[1]

        replay_messages.append(messages[msg_idx + 1])
        round_text = tokenizer.apply_chat_template(replay_messages, tokenize=False)
        round_encoding = tokenizer(
            round_text,
            return_tensors="pt",
            return_special_tokens_mask=True,
        ).to(device)
        round_len = round_encoding.input_ids.shape[1]
        suffix_ids = round_encoding.input_ids[0, prompt_len:round_len]
        suffix_special = round_encoding.special_tokens_mask[0, prompt_len:round_len].bool()
        suffix_mask = ~suffix_special
        if tokenizer.eos_token_id is not None:
            suffix_mask = suffix_mask & (suffix_ids != tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            suffix_mask = suffix_mask & (suffix_ids != tokenizer.pad_token_id)
        action_mask[prompt_len:round_len] = suffix_mask
        msg_idx += 2

    return sequence_ids, action_mask

def eval_probe(game, tokenizer, model):
    device = model.device
    messages = [
        {"role": "user", "content": _user_turn_content(game, include_instructions=True)},
    ]
    messages_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    messages_encoding = tokenizer(messages_text, return_tensors="pt").to(device)
    candidates, action_scores = _score_action_candidates(
        messages_encoding,
        tokenizer,
        model,
    )
    action_probs = F.softmax(action_scores, dim=0)
    by_action = {
        candidate["action"]: action_probs[idx].item()
        for idx, candidate in enumerate(candidates)
    }
    return {
        "p_hit": by_action.get("HIT", 0.0),
        "p_stay": by_action.get("STAY", 0.0),
    }


def generate_trajectory(game, tokenizer, model, temperature: float = 1.0):
    device = model.device
    messages = []

    invalid_action = False
    #print("start game")
    while not game.get_state()["game_over"]:
        state_str = _user_turn_content(game, include_instructions=not messages)
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
        generate_kwargs = {
            "max_new_tokens": 4,
            "pad_token_id": tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
        }
        if temperature <= 0.0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        with torch.no_grad():
            outputs = model.generate(
                **messages_encoding,
                **generate_kwargs,
            )
        response_ids = outputs.sequences[0, prompt_len:]
        action = tokenizer.batch_decode(response_ids.unsqueeze(0), skip_special_tokens=True)[0]
        action = action.strip().upper()
        #print(f"Action: {action}")
        messages.append({ "role": "assistant", "content": action})
        #print(action)

        if action == "HIT":
            #print("model hits")
            observation = game.hit()
        elif action == "STAY":
            #print("model stays")
            observation = game.stand()
        else:
            #print("invalid action")
            invalid_action = True
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

    sequence_ids, action_mask = _build_training_sequence(tokenizer, messages, device)
    token_rewards = torch.zeros(sequence_ids.shape[0], dtype=torch.float32, device=device)
    token_rewards[action_mask] += reward

    #print("Reward:", reward)
    #print("Mask:", action_mask)

    with torch.no_grad():
        sequence_attention_mask = torch.ones_like(sequence_ids.unsqueeze(0))
        logits = model(
            sequence_ids.unsqueeze(0),
            attention_mask=sequence_attention_mask,
        ).logits
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
