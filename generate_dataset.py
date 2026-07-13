import csv
import random
import os

random.seed(42)

positive_templates = [
    "This product is amazing and I love it.",
    "I absolutely love this product.",
    "This is the best thing I have ever bought.",
    "Fantastic quality, very impressed!",
    "Amazing product, exceeded my expectations.",
    "I am so happy with my purchase.",
    "This exceeded all my expectations.",
    "I could not be happier with this.",
    "Wow, this is absolutely incredible!",
    "This product is fantastic.",
    "I am very impressed with the quality.",
    "What an amazing product!",
    "This works perfectly, highly recommend.",
    "I am extremely satisfied with this.",
    "Best purchase I have made this year.",
]

positive_continuations = [
    " The build quality is superb.",
    " It works exactly as described.",
    " It was very easy to set up.",
    " The material is high quality.",
    " It is very well made and durable.",
    " It looks even better in person.",
    " The customer service was excellent.",
    " The shipping was fast and free.",
    " It performs better than expected.",
    " The features are exactly what I wanted.",
    " It is very user friendly.",
    " It is comfortable and well designed.",
]

positive_endings = [
    " Worth every penny.",
    " Would definitely buy again.",
    " Could not ask for more.",
    " Highly recommended.",
    " Best purchase this year.",
]

negative_templates = [
    "This is the worst product I have ever used.",
    "I absolutely hate this product.",
    "This product is garbage.",
    "What a waste of money.",
    "This is a total disappointment.",
    "I wish I could get my money back.",
    "This product is terrible.",
    "Do not waste your money on this.",
    "I am extremely unhappy with this.",
    "This is absolutely terrible.",
    "Complete waste of money.",
    "This product is a scam.",
]

negative_continuations = [
    " It stopped working after a week.",
    " The quality is very poor and cheap.",
    " It feels very cheap and flimsy.",
    " It arrived damaged and broken.",
    " It broke the first time I used it.",
    " It is very uncomfortable to use.",
    " The buttons stopped working immediately.",
    " Delivery took much longer than promised.",
    " It does not work as advertised.",
    " It fell apart within days.",
]

negative_endings = [
    " I will never buy from this brand again.",
    " Save your money.",
    " Do not buy this.",
    " I am returning it immediately.",
    " I want a full refund.",
]

neutral_templates = [
    "The product arrived yesterday.",
    "It is okay, nothing special.",
    "The product is fine.",
    "This is an average product.",
    "The product works as expected.",
    "I have mixed feelings about this.",
    "I have no strong feelings about this.",
    "The product is just okay.",
    "This is neither good nor bad.",
    "Not bad, not great either.",
    "The product is decent.",
    "It functions as intended.",
    "This product is adequate.",
    "The product does what it is supposed to do.",
    "I am neutral about this product.",
]

neutral_continuations = [
    " It gets the job done.",
    " The quality is acceptable for the price.",
    " It is comparable to other products in this price range.",
    " It matches the description.",
    " The packaging was standard.",
    " It meets basic expectations.",
    " It is what I expected for the price.",
]

neutral_endings = [
    " It will do.",
    " It is fine for now.",
    " I have no complaints.",
    " Nothing to write home about.",
]

def make_review(templates, continuations, endings):
    parts = [random.choice(templates)]
    if random.random() < 0.4:
        parts.append(random.choice(continuations))
    if random.random() < 0.3:
        parts.append(random.choice(endings))
    return "".join(parts).strip()

rows = []

for _ in range(1000):
    rows.append((make_review(positive_templates, positive_continuations, positive_endings), "Positive"))
    rows.append((make_review(negative_templates, negative_continuations, negative_endings), "Negative"))
    rows.append((make_review(neutral_templates, neutral_continuations, neutral_endings), "Neutral"))

# Duplicate sample reviews many times
samples = [
    ("This product is amazing and I love it.", "Positive"),
    ("The product arrived yesterday.", "Neutral"),
    ("This is the worst product I have ever used.", "Negative"),
]
for _ in range(50):
    for text, label in samples:
        rows.append((text, label))

# Duplicate the key missing phrase
for _ in range(50):
    rows.append(("Fantastic quality, very impressed!", "Positive"))
    rows.append(("Fantastic quality, highly recommend!", "Positive"))
    rows.append(("Amazing product, very impressed.", "Positive"))
    rows.append(("I am very impressed with the quality.", "Positive"))
    rows.append(("What a fantastic item.", "Positive"))

random.shuffle(rows)

os.makedirs("dataset", exist_ok=True)
with open("dataset/sentiment_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "sentiment"])
    writer.writerows(rows)

counts = {}
for _, s in rows:
    counts[s] = counts.get(s, 0) + 1
print(f"Generated {len(rows)} reviews")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")
