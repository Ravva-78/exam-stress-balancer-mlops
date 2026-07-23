import requests
import json
import time
import sys

url = "https://exam-stresss-balancer.onrender.com/api/explain"

inputs = [
  {"stress_level":"low","hours_studied":2,"days_until_exam":30,"current_performance":0.8},
  {"stress_level":"low","hours_studied":8,"days_until_exam":1,"current_performance":0.9},
  {"stress_level":"medium","hours_studied":4,"days_until_exam":7,"current_performance":0.5},
  {"stress_level":"medium","hours_studied":0,"days_until_exam":14,"current_performance":0.3},
  {"stress_level":"high","hours_studied":6,"days_until_exam":3,"current_performance":0.4},
  {"stress_level":"high","hours_studied":10,"days_until_exam":1,"current_performance":0.2},
  {"stress_level":"critical","hours_studied":0,"days_until_exam":1,"current_performance":0.1},
  {"stress_level":"critical","hours_studied":8,"days_until_exam":5,"current_performance":0.6},
  {"stress_level":"low","hours_studied":0,"days_until_exam":0,"current_performance":0.0},
  {"stress_level":"critical","hours_studied":24,"days_until_exam":30,"current_performance":1.0}
]

print("Waiting for deployment to go live...")
# We will poll with the first input until it stops returning 422
max_retries = 30
deployed = False
for attempt in range(max_retries):
    try:
        res = requests.post(url, json=inputs[0])
        if res.status_code == 200:
            print("Deployment is LIVE! Schema mapping is working.")
            deployed = True
            break
        elif res.status_code == 422:
            print(f"Still returning 422. Waiting... ({attempt+1}/{max_retries})")
        else:
            print(f"Unexpected status: {res.status_code}. Waiting... ({attempt+1}/{max_retries})")
    except Exception as e:
        print(f"Error connecting: {e}. Waiting...")
    time.sleep(10)

if not deployed:
    print("Deployment didn't go live within 5 minutes.")
    sys.exit(1)

print("\n--- Final Reliability Evaluation ---")
schema_pass_count = 0
for i, data in enumerate(inputs):
    try:
        res = requests.post(url, json=data)
        if res.status_code == 200:
            schema_pass_count += 1
        print(f"\n--- Input {i} ---")
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(0.5)

print(f"\nSchema Integrity Passes: {schema_pass_count}/{len(inputs)}")

print("\n--- Consistency Test ---")
test_input = inputs[2]
for i in range(3):
    try:
        res = requests.post(url, json=test_input)
        print(f"Run {i+1} Response: {res.text}")
    except Exception as e:
        print(f"Run {i+1} Error: {e}")
    time.sleep(0.5)
