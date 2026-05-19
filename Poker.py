import cv2
import numpy as np
from ultralytics import YOLO
from treys import Card, Evaluator, Deck

# --- CONFIG ---
MODEL_PATH = r"C:\Users\user\Downloads\best.pt"
model = YOLO(MODEL_PATH)
evaluator = Evaluator()

is_frozen = False
frozen_info = {"equity": 0, "my_type": "", "opp_type": ""}
saved_frame = None

def calculate_h2h_equity(my_hand, opp_hand, board, iterations=2000):
    """Calculates win % between two specific hands"""   
    wins = 0
    for _ in range(iterations):
        deck = Deck()
        all_known = my_hand + opp_hand + board
        deck.cards = [c for c in deck.cards if c not in all_known]
        
        # Draw remaining board cards (if any)
        remaining = deck.draw(5 - len(board))
        full_board = board + remaining
        
        p_score = evaluator.evaluate(full_board, my_hand)
        o_score = evaluator.evaluate(full_board, opp_hand)
        
        if p_score < o_score: wins += 1
        elif p_score == o_score: wins += 0.5
            
    return round((wins / iterations) * 100, 1)

def clean_label(label):
    if len(label) < 2: return None
    rank = label[:-1].replace("10", "T")
    suit = label[-1].lower()
    return f"{rank}{suit}"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    if not is_frozen:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # --- THE FLIP TRICK ---
        # We create a version of the frame where the top 30% is rotated 180 degrees
        # This helps the AI read the opponent's upside-down cards
        analysis_frame = frame.copy()
        top_zone_limit = int(h * 0.3)
        analysis_frame[0:top_zone_limit, :] = cv2.rotate(analysis_frame[0:top_zone_limit, :], cv2.ROTATE_180)
        
        results = model(analysis_frame, conf=0.35, iou=0.3, verbose=False)
        display_frame = frame.copy()
    else:
        display_frame = saved_frame.copy()

    temp_player, temp_opp, temp_board = [], [], []
    seen = set()

    if not is_frozen and results[0].boxes:
        for box in sorted(results[0].boxes, key=lambda x: x.conf, reverse=True):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            c_str = clean_label(label)

            if c_str and c_str not in seen:
                seen.add(c_str)
                c_obj = Card.new(c_str)
                
                # 30/40/30 Split Logic
                if y1 < (h * 0.3): # TOP 30% (Opponent)
                    if len(temp_opp) < 2: temp_opp.append(c_obj)
                elif y1 > (h * 0.7): # BOTTOM 30% (Player)
                    if len(temp_player) < 2: temp_player.append(c_obj)
                else: # MIDDLE 40% (Board)
                    if len(temp_board) < 5: temp_board.append(c_obj)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(display_frame, label, (x1, y1-5), 0, 0.4, (0, 255, 0), 1)

    # --- UI LAYOUT ---
    # Zone Lines
    cv2.line(display_frame, (0, int(h*0.3)), (w, int(h*0.3)), (255, 255, 0), 1) # Opponent Line
    cv2.line(display_frame, (0, int(h*0.7)), (w, int(h*0.7)), (255, 0, 0), 1)   # Player Line

    # Data Panel
    cv2.rectangle(display_frame, (5, h-120), (450, h-5), (0, 0, 0), -1)
    
    if is_frozen:
        txt = f"YOU: {frozen_info['equity']}% Win | {frozen_info['my_type']}"
        cv2.putText(display_frame, txt, (15, h-40), 0, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "FROZEN - Press Space to live", (15, h-90), 0, 0.5, (0, 255, 255), 1)
    else:
        status = f"Opp:{len(temp_opp)}  Board:{len(temp_board)}  You:{len(temp_player)}"
        color = (0, 255, 0) if (len(temp_opp)==2 and len(temp_player)==2 and len(temp_board)>=3) else (0,0,255)
        cv2.putText(display_frame, status, (15, h-40), 0, 0.7, color, 2)
        cv2.putText(display_frame, "LIVE - Press Space to Compare", (15, h-90), 0, 0.5, color, 1)

    cv2.imshow("Poker AI - 30/40/30 Head-to-Head", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32: # SPACE
        if not is_frozen and len(temp_opp)==2 and len(temp_player)==2 and len(temp_board)>=3:
            eq = calculate_h2h_equity(temp_player, temp_opp, temp_board)
            # Get current hand types
            my_t = evaluator.class_to_string(evaluator.get_rank_class(evaluator.evaluate(temp_board, temp_player)))
            frozen_info = {"equity": eq, "my_type": my_t}
            saved_frame = display_frame.copy()
            is_frozen = True
        else:
            is_frozen = False
    elif key == ord('q'): break


cap.release()
cv2.destroyAllWindows()
