import random


def model_eval(S_u, S_b, q, k):
    result = random.choices([0, 1], weights=[1.0 - q, q])[0]

    if result == 0:
        return {
            'wins': 'NONE',
            'W_u': 0,
            'W_b': 0
        }

    P_u = S_u / (S_u + k * S_b)
    P_b = (k * S_b) / (S_u + k * S_b)

    W = S_u + S_b
    W_u = W - S_u
    W_b = W - S_b

    result = random.choices([0, 1], weights=[P_u, P_b])[0]

    wins = ''
    if result == 1:  # Bot wins
        wins = 'BOT'
        W_u = -W_u
    else:
        wins = 'USER'
        W_b = -W_b

    return {
        'wins': wins,
        'W_u': W_u,
        'W_b': W_b
    }


# Main Model parameters
k = 2.0
q_min = 0.44
q_max = 0.85
r_min = 0.2
r_max = 2.5
#######################

quantity = 10000000
balance = 1000000

total = {
    'W_u': 0,
    'W_b': 0,
    'bot_wins': 0,
    'user_wins': 0,
    'user_balance': balance,
    'bot_balance': balance,
}

for i in range(quantity):
    S_u = round(random.uniform(1, 2), 2)
    r = S_u * random.uniform(r_min, r_max)
    S_b = round(r, 2)
    q = round(random.uniform(q_min, q_max), 2)

    #print(f'Game {i + 1} / {quantity} : S_umax = {S_umax}, S_bmax = {S_bmax}')
    result = model_eval(S_u, S_b, q, k)

    if result['wins'] == 'BOT':
        total['bot_wins'] += 1
        total['bot_balance'] += round(result['W_b'], 2)
        total['user_balance'] -= round(S_u, 2)
    elif result['wins'] == 'USER':
        total['user_wins'] += 1
        total['bot_balance'] -= round(S_b, 2)
        total['user_balance'] += round(result['W_u'], 2)
    else:
        total['user_wins'] += 1

    total['W_u'] += result['W_u']
    total['W_b'] += result['W_b']

# Print totals
print("--- TOTAL ---")
print("Total games: ", quantity)
print("Bot wins: ", total['bot_wins'])
print("User wins: ", total['user_wins'])
print("User income: ", total['W_u'])
print("Bot income: ", total['W_b'])
print("Bot account: ", round(total['bot_balance'], 2))
print("User account: ", round(total['user_balance'], 2))
print(f"Bot wins percent: {round(total['bot_wins'] / quantity * 100.0, 2)}%")
