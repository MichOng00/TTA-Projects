import random, time
from difflib import SequenceMatcher
from IPython.display import clear_output

real_string = ""
def add_hidden_chars(text):
    hidden_char = "\u200B"  # Zero Width Space
    return hidden_char.join(text)  # Inserts between every character

# Function to detect cheating by checking hidden characters
def detect_cheating(user_input):
    hidden_char = "\u200B"
    return hidden_char in user_input  # Returns True if cheating detected


def get_rank(score):
    ranks = [
        (0, "❌ Unranked ❌ (0-9)"), (10, "🐢 Slow Starter 🐢 (10-12)"), (13, "⌨️ Basic Typer ⌨️ (13-15)"),
        (16, "📖 Typing Novice 📖 (16-18)"), (19, "🎓 Keyboard Rookie 🎓 (19-21)"), (22, "🍎 Typing Trainee 🍎 (22-24)"),
        (25, "🔥 Speed Seeker 🔥 (25-27)"), (28, "🚀 Typing Explorer 🚀 (28-30)"), (31, "🤹 Quick Hands 🤹 (31-33)"),
        (34, "🏋️ Rapid Typer 🏋️ (34-36)"), (37, "🏃‍♂️ Keyboard Sprinter 🏃‍♂️ (37-39)"), (40, "🛡️ Fast Learner 🛡️ (40-43)"),
        (44, "🎩 Typing Challenger 🎩 (44-47)"), (48, "⚔️ Keyboard Warrior ⚔️ (48-51)"), (52, "✈️ Wingy Writer ✈️ (52-55)"),
        (56, "🎸 Hyper Typer 🎸 (56-59)"), (60, "🏁 Speed Demon 🏁 (60-64)"), (65, "👑 Elite Typer 👑 (65-69)"),
        (70, "🏎️ Ultra Typer 🏎️ (70-74)"), (75, "🌪️ Tornado Typist 🌪️ (75-79)"), (80, "⚡ Lightning Fingers ⚡ (80-84)"),
        (85, "🏆 Supreme Keyboarder 🏆 (85-89)"), (90, "🎯 Ultimate Typist 🎯 (90-94)"), (95, "🎹 Master of Keys 🎹 (95-99)"),
        (100, "👾 Typing Legend 👾 (100-119)"), (120, "🤓 Mr. Rafa 🤓 (120+)")
    ]

    current_rank = ranks[0][1]
    for threshold, rank in ranks:
        if score >= threshold:
            current_rank = rank
        else:
            break

    return current_rank



# Function to read highscore off a .txt file
def load_high_scores():
    highscores = {"easy": 0, "normal": 0, "hard": 0}
    filename = "highscores.txt"
    
    try:
        with open(filename, "r") as file:
            for line in file:
                difficulty, score = line.strip().split(":")
                highscores[difficulty] = float(score)
    except FileNotFoundError:
        with open(filename, "w") as file:
            for difficulty in highscores:
                file.write(f"{difficulty}:0\n")  # Initialize scores if file not found

    return highscores

def save_high_scores(scores):
    filename = "highscores.txt"
    with open(filename, "w") as file:
        for difficulty, score in scores.items():
            file.write(f"{difficulty}:{score}\n")
    print("High score successfully saved!")

# Function to generate a random sequence of words from the selected difficulty level
def generate_word_string(word_dict, difficulty, num_words=10):
    global real_string
    target_string = " ".join(random.sample(word_dict[difficulty], num_words))
    real_string = target_string
    hidden_string = add_hidden_chars(target_string)  # Add hidden characters
    return(hidden_string)
    

# Function to calculate accuracy based on character similarity
def calculate_accuracy(correct_string, user_input):
    matcher = SequenceMatcher(None, correct_string, user_input)
    accuracy = matcher.ratio() * 100  # Returns a percentage
    return accuracy


# Score modifiers based on difficulty
difficulty_multipliers = {
    "easy": 1.0,
    "normal": 1.6,
    "hard": 2.3
}

# Function to calculate the final score based on WPM and accuracy
def calculate_score(wpm, accuracy, bonus, difficulty):
    base_score = wpm
    modifier = accuracy / 100
    difficulty_multiplier = difficulty_multipliers[difficulty]  # Get the multiplier
    
    final_score = round(bonus + wpm * modifier * difficulty_multiplier, 1)
    return final_score

def typing_game():
    bonus = 0
    highscore = load_high_scores()  # Variable to track the highest score
    
    # Dictionary containing different difficulty levels with respective words
    word_dict = {
    "easy": [
        'cat', 'dog', 'car', 'box', 'sky', 'top', 'ply', 'rat', 'pig', 'fit',
        'jog', 'run', 'mum', 'dad', 'pin', 'far', 'war', 'bar', 'wet', 'try',
        'pry', 'cry', 'sad', 'mix', 'put', 'for', 'cut', 'tax', 'pot', 'zoo',
        'tap', 'lot', 'hat', 'bit', 'tan', 'tag', 'yet', 'jaw', 'jet', 'joy',
        'cue', 'dim', 'egg', 'eye', 'ice', 'ink', 'ham', 'has', 'goo', 'map',
        'lab', 'key', 'ski', 'sun', 'saw', 'paw', 'use', 'zap', 'wag', 'yes',
        'pal', 'new', 'one', 'awe', 'dry', 'cop', 'bun', 'elk', 'ear', 'our',
        'mad', 'max', 'fan', 'any', 'vet', 'urn', 'set', 'pad', 'net', 'tin',
        'vow', 'led', 'mud', 'sap', 'keg', 'art', 'oar', 'ill', 'pop', 'sat',
        'van', 'own', 'not', 'yak', 'yam', 'imp', 'toy', 'pan', 'kin', 'owl',
        'four', 'play', 'stop', 'book', 'lamp', 'fish', 'milk', 'star', 'cake', 'home',
        'lion', 'bird', 'gate', 'door', 'coin', 'desk', 'snow', 'fire', 'leaf', 'boat',
        'moon', 'bear', 'frog', 'duck', 'ring', 'park', 'rock', 'hawk', 'road', 'bike',
        'ball', 'shoe', 'wind', 'rain', 'sand', 'wave', 'game', 'rice', 'bean', 'rose',
        'corn', 'pear', 'lock', 'belt', 'kite', 'nest', 'time', 'path', 'card', 'drum',
        'sign', 'seed', 'flag', 'tent', 'king', 'zero', 'peak', 'pool', 'coat', 'nose',
        'pump', 'rope', 'bush', 'ship', 'soap', 'tank', 'sock', 'tube', 'bell', 'wall',
        'cart', 'bake', 'jump', 'calm', 'yard', 'oven', 'clay', 'deer', 'mine', 'lava',
        'step', 'tree', 'tape', 'wolf', 'page', 'wire', 'data', 'crop', 'film', 'fuel',
        'plug', 'meal', 'palm', 'hill', 'date', 'echo', 'herb', 'mile', 'song', 'cave'
    ], 
    "normal": [
        'apple', 'bread', 'chair', 'dream', 'eagle', 'field', 'glass', 'honey', 'igloo', 'jelly',
        'knife', 'lemon', 'magic', 'night', 'ocean', 'paper', 'queen', 'river', 'snake', 'table',
        'uncle', 'valve', 'wheel', 'youth', 'zebra', 'angel', 'beach', 'crane', 'dance', 'earth',
        'flute', 'giant', 'hotel', 'image', 'juice', 'koala', 'laser', 'metal', 'novel', 'olive',
        'panda', 'quiet', 'robot', 'stone', 'tiger', 'urban', 'vivid', 'wagon', 'xenon', 'young',
        'zesty', 'bloom', 'cloud', 'drill', 'ember', 'fable', 'grape', 'heart', 'index', 'judge',
        'kiosk', 'light', 'mango', 'north', 'onion', 'plant', 'quilt', 'radio', 'solar', 'toast',
        'umbra', 'vapor', 'wheat', 'xylem', 'yeast', 'zonal', 'spoon', 'timer', 'piano', 'brush',
        'storm', 'baker', 'camel', 'diary', 'frost', 'goose', 'haven', 'inbox', 'joint', 'kayak',
        'lodge', 'motel', 'nerve', 'orbit', 'pearl', 'query', 'ranch', 'salad', 'theme', 'virus',
        'animal', 'butter', 'castle', 'doctor', 'effort', 'forest', 'garden', 'hammer', 'island', 'jungle',
        'kitten', 'ladder', 'mother', 'nature', 'orange', 'pencil', 'quartz', 'rocket', 'silver', 'ticket',
        'urgent', 'valley', 'window', 'yellow', 'zipper', 'anchor', 'bakery', 'carpet', 'donkey', 'energy',
        'falcon', 'galaxy', 'helmet', 'icicle', 'jacket', 'kidney', 'lizard', 'marble', 'nickel', 'octave',
        'palace', 'quiver', 'ribbon', 'shadow', 'tomato', 'united', 'velvet', 'wallet', 'yogurt', 'zigzag',
        'barrel', 'candle', 'dagger', 'engine', 'flower', 'glider', 'harbor', 'insect', 'jumper', 'kettle',
        'locker', 'magnet', 'napkin', 'orchid', 'pillow', 'quaint', 'refuge', 'sphere', 'throat', 'umpire',
        'vacuum', 'walnut', 'street', 'zodiac', 'avenue', 'beacon', 'cactus', 'desert', 'fossil', 'guitar',
        'honest', 'impact', 'jogger', 'karate', 'lawyer', 'meteor', 'nectar', 'oxygen', 'poetry', 'remedy',
        'saddle', 'teapot', 'utopia', 'violin', 'wizard', 'yawner', 'zephyr', 'arctic', 'ballot', 'cheese'
    ],  
    "hard": [
        'airport', 'battery', 'cabinet', 'diamond', 'evening', 'factory', 'gallery', 'harvest', 'imagine', 'journal',
        'kitchen', 'library', 'morning', 'natural', 'orchard', 'pumpkin', 'quarter', 'rainbow', 'stadium', 'teacher',
        'unicorn', 'village', 'weather', 'perfect', 'zealous', 'balcony', 'captain', 'dentist', 'emerald', 'freedom',
        'giraffe', 'holiday', 'inquiry', 'justice', 'kingdom', 'lantern', 'machine', 'network', 'octopus', 'penguin',
        'quality', 'railway', 'sandals', 'tornado', 'upgrade', 'vitamin', 'windows', 'yawning', 'zipline', 'athlete',
        'bargain', 'concert', 'dessert', 'ecology', 'fortune', 'gateway', 'harmony', 'inspect', 'journey', 'keyword',
        'laundry', 'mineral', 'notepad', 'opinion', 'passion', 'quilted', 'receipt', 'sausage', 'tribute', 'utensil',
        'variety', 'warrior', 'younger', 'feather', 'advance', 'blanket', 'curtain', 'dolphin', 'earring', 'trainer',
        'glacier', 'hamster', 'isolate', 'jackpot', 'kneecap', 'lobster', 'monster', 'numeric', 'origami', 'pancake',
        'recycle', 'sandbox', 'texture', 'utility', 'vaccine', 'whistle', 'yearned', 'zoology', 'amplify', 'bouquet',
        'elephant', 'mountain', 'computer', 'hospital', 'sunshine', 'painting', 'railroad', 'sandwich', 'umbrella', 'vacation',
        'baseball', 'calendar', 'dinosaur', 'engineer', 'festival', 'gasoline', 'hardware', 'illusion', 'junction', 'kangaroo',
        'language', 'magazine', 'neighbor', 'ornament', 'passport', 'question', 'reindeer', 'scissors', 'teaspoon', 'laziness',
        'wildlife', 'surprise', 'youthful', 'zeppelin', 'aquarium', 'bluebird', 'cucumber', 'driveway', 'election', 'firework',
        'goldfish', 'headache', 'industry', 'jalapeno', 'knitting', 'laughter', 'landmark', 'kangaroo', 'magician', 'opposite',
        'football', 'reindeer', 'scissors', 'teaspoon', 'vineyard', 'airfield', 'birthday', 'cinnamon', 'dialogue', 'envelope',
        'fraction', 'giftwrap', 'homeland', 'infinite', 'jeweller', 'kangaroo', 'language', 'magnetic', 'nickname', 'ornament',
        'passport', 'romantic', 'sapphire', 'tortoise', 'ultimate', 'velocity', 'workshop', 'yachting', 'zucchini', 'airplane',
        'birthday', 'cinnamon', 'dialogue', 'eggplant', 'flagship', 'graceful', 'handsome', 'ironical', 'software', 'knapsack',
        'landmark', 'mandolin', 'medieval', 'nutrient', 'opponent', 'polished', 'doorbell', 'campfire', 'backyard', 'flagship'
    ]  
}
    
    print("Welcome to TypeTester! Choose your level by typing its difficulty.")
    highscores = load_high_scores()

    for difficulty, score in highscores.items():
        print(f"{difficulty.capitalize()} - Highest score: {score}, Rank: {get_rank(score)}")
    difficulty = input("Easy (3-4 letter words),  Normal (5-6 letter words),  Hard (7-8 letter words), Quit (Exits the game)\n").strip().lower()
    
    while True:
        if difficulty == "quit":  # Exit condition
            break
        elif difficulty not in word_dict:
            print("Invalid difficulty! Try again.")
            difficulty = input().strip().lower()
        else:
            clear_output(wait=True)
            target_string = generate_word_string(word_dict, difficulty)  # Generate words to type
            print("Type this as fast as you can:\n")
            print(target_string, end='\n')
            print("\nGet ready!")
            time.sleep(3)
            # Countdown before starting
            for x in range(3):
                print(3-x, end='\r')
                time.sleep(1)
            print("Go!")
            
            start_time = time.time()  # Start timer
            user_input = input("\nYour input: ")

            # Anti-cheat detection
            if detect_cheating(user_input):
                print("\n⚠️ Cheating detected! Copy-pasting is not allowed.")
                time.sleep(3)
                difficulty = input("Easy (3-4 letter words),  Normal (5-6 letter words),  Hard (7-8 letter words),  Quit (Exits the game)\n").strip().lower()
                continue
            duration = time.time() - start_time  # Calculate time taken
            
            if duration < 1:  # Prevent unrealistically high WPM
                duration = 1
            
            accuracy = calculate_accuracy(real_string, user_input)  # Calculate accuracy
            
            # Check if user typed correctly
            if user_input == target_string:
                print("\nPerfect! You typed everything correctly!(+15 bonus points)")
            elif accuracy < 75:
                print("\nNice try! You weren't so accurate though...")
            elif accuracy < 90:
                print("\nGood try! There is still room for improvement though!")
            elif accuracy < 95:
                print("\nGreat try! You only made a few mistakes!(+5 bonus points)")
            elif accuracy < 100:
                print("\nAwesome try! You almost typed it perfectly!(+10 bonus points)")
            
            # Calculate words per minute
            wpm = 10 * (60 / duration)
            
            # Display results
            print(f"Time taken: {duration:.2f} seconds")
            print(f"Words per minute: {wpm:.1f} wpm")
            print(f"Accuracy: {accuracy:.2f}%")
            if accuracy == 100:
                bonus = 15  # Perfect string bonus
            elif accuracy < 90:
                pass
            elif accuracy < 95:
                bonus = 5
            elif accuracy < 100:
                bonus = 10
            score = calculate_score(wpm, accuracy, bonus, difficulty)  # Calculate score
            
            if accuracy < 75:
                print("Score: Invalid! Accuracy too low!")
            elif score > highscores[difficulty]:
                print("Score:", score, "\nNew high score! Congratulations!")
                print(f"Rank: {get_rank(score)}")
                highscores[difficulty] = score
                save_high_scores(highscores)
            else:
                print("Score:", score)
                print(f"Rank: {get_rank(score)}")
            
            time.sleep(5)
            print("\nPlay again?")
            for difficulty, score in highscores.items():
                print(f"{difficulty.capitalize()} - Highest score: {score}, Rank: {get_rank(score)}")
            time.sleep(2)
            difficulty = input("Easy (3-4 letter words),  Normal (5-6 letter words),  Hard (7-8 letter words),  Quit (Exits the game)\n").strip().lower()

typing_game()