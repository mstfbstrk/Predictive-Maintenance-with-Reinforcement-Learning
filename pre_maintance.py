import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random

# --- 1. ORTAM ---
class MaintenanceEnv(gym.Env):
    def __init__(self, csv_path):
        super(MaintenanceEnv, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(2)
        self.current_idx = 0
        self.state = 0

    def composite_state(self, row):
        temp_diff = row['Process temperature [K]'] - row['Air temperature [K]']
        torque = row['Torque [Nm]']
        wear = row['Tool wear [min]']
        
        if row['Machine failure'] == 1 or temp_diff > 11.0 or torque > 60 or wear > 200:
            return 3 # Kritik
        elif temp_diff > 10.0 or torque > 50 or wear > 150:
            return 2 # Uyarı
        elif temp_diff > 8.5 or wear > 80:
            return 1 # Normal
        return 0     # Yeni

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = random.randint(0, len(self.df) - 300)
        self.state = self.composite_state(self.df.iloc[self.current_idx])
        return self.state, {}

    def step(self, action):
        done = False
        
        if action == 1: # BAKIM YAP
            reward = -20  # Bakım maliyeti
            self.state = 0 # Makineyi 'Yeni' durumuna döndür
        else: # DEVAM ET
            self.current_idx += 1
            if self.current_idx >= len(self.df):
                return self.state, 0, True, False, {}
                
            row = self.df.iloc[self.current_idx]
            self.state = self.composite_state(row)
            
            if row['Machine failure'] == 1:
                reward = -1000 # Bozulması durumunda alınan büyük ceza
                done = True   # Makine bozuldu
            else:
                reward = 50   # Güvenli çalışma ödülü
        
        return self.state, reward, done, False, {}

# --- 2. EĞİTİM ---
env = MaintenanceEnv('ai4i2020.csv')
q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9998 # Her bölümde epsilon'ı yavaşça azaltarak keşif oranı düşürüldü
epsilon_min = 0.01
episodes = 25000

all_rewards = []
all_steps = []

print("Eğitim Başlatılıyor...")
for i in range(1, episodes + 1):
    state, _ = env.reset()
    episode_reward = 0
    steps = 0
    done = False

    while not done and steps < 150: 
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)
        
        # Bellman Denklemi ile Q-table güncellemesi
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        episode_reward += reward
        steps += 1
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    all_rewards.append(episode_reward)
    all_steps.append(steps)
    
    if i % 5000 == 0: print(f"Bölüm {i}/{episodes} tamamlandı. Epsilon: {epsilon:.2f}")

# --- 3. GÖRSELLEŞTİRME VE ANALİZ ---
def moving_average(data, window_size=500):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(all_rewards, alpha=0.1, color='blue')
plt.plot(moving_average(all_rewards), color='darkblue', linewidth=2)
plt.title('Toplam Ödül Eğilimi')
plt.ylabel('Puan')

plt.subplot(1, 2, 2)
plt.plot(all_steps, alpha=0.1, color='orange')
plt.plot(moving_average(all_steps), color='red', linewidth=2)
plt.title('Bozulana Kadar Geçen Süre')
plt.ylabel('Adım')
plt.tight_layout()
plot_file = 'egitim_grafikleri.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Done! 'egitim_grafikleri.png' saved.")
plt.show()

# --- 4. SONUÇLAR ---
print("\n--- ÖĞRENİLEN Q-TABLE ---")
print("Durum\t\tDevam Et\tBakım Yap")
labels = ["Yeni", "Normal", "Uyarı", "Kritik"]
for i, label in enumerate(labels):
    print(f"{label:10}\t{q_table[i,0]:.2f}\t\t{q_table[i,1]:.2f}")

print("\n--- OPTİMAL POLİTİKA ---")
for i, label in enumerate(labels):
    best = "Devam Et" if q_table[i,0] > q_table[i,1] else "Bakım Yap"
    print(f"{label:10} -> {best}")