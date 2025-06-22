import pandas as pd

# Simulated text messages
texts = [
    "Win a free vacation now",
    "Exclusive offer just for you",
    "Click to claim your free prize",
    "Limited time deal, act fast",
    "You have been selected for a gift",
    "Are we still meeting today?",
    "Let's grab lunch tomorrow",
    "Can you send me the report?",
    "Don't forget the meeting at 3 PM",
    "I will call you in 10 minutes",
    "Congratulations! You've won a car",
    "Urgent: Verify your bank account",
    "Your OTP is 4531",
    "How was your presentation?",
    "I'll be late, stuck in traffic"
]

# Corresponding labels
labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Ham", "Ham", "Ham", "Ham", "Ham",
    "Spam", "Spam", "Ham", "Ham", "Ham"
]

# Create DataFrame
df = pd.DataFrame({
    "text": texts,
    "label": labels
})

# Save to CSV
df.to_csv("dataset-for-multinomial-nb.csv", index=False)

# Preview
# print(df.head())
