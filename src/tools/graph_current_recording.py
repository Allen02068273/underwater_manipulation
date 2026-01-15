import pandas as pd
import matplotlib.pyplot as plt

csv_file = "data/current_recordings/current_20260115_131719.csv"
# csv_file = "data/current_recordings/current_20260115_131918.csv"
df = pd.read_csv(csv_file)

time = df.index.to_numpy()

columns = ['Gripper', 'Joint B', 'Joint C', 'Joint D', 'Base Joint']

plt.figure(figsize=(12, 6))
for col in columns:
    plt.plot(time, df[col].to_numpy(), label=col)

plt.xlabel('Time (samples)')
plt.ylabel('Current')
plt.title('Joint Currents Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
