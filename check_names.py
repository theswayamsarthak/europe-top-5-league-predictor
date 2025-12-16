import pandas as pd
import io
import requests

# Check Germany (D1)
url = "https://www.football-data.co.uk/mmz4281/2425/D1.csv"
print("Downloading D1 Data...")
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('latin-1')))

print("\n--- TEAMS IN CSV FILE ---")
teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique())
for t in teams:
    print(f"'{t}'")