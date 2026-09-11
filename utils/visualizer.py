import cv2
import numpy as np

class HUDVisualizer:
    """
    Renders a modern, professional surveillance HUD overlay for Quarrel Detection AI.
    """
    # Color palette (BGR)
    COLOR_BG_DARK = (20, 20, 25)
    COLOR_GREEN = (46, 204, 113)
    COLOR_YELLOW = (41, 128, 185) # Amber/Yellow
    COLOR_AMBER = (0, 165, 255)
    COLOR_RED = (46, 46, 235)
    COLOR_WHITE = (245, 245, 245)
    COLOR_GRAY = (140, 140, 140)
    COLOR_CYAN = (255, 215, 0)

    def __init__(self):
        self.flash_counter = 0

    def draw_hud(self, frame, eval_result, tracked_faces, fps, audio_status, source_name="Camera"):
        h, w = frame.shape[:2]
        self.flash_counter += 1

        state = eval_result.get('state', 'NORMAL')
        risk_score = eval_result.get('smoothed_risk', 0.0)
        quarrel_confirmed = eval_result.get('quarrel_confirmed', False)

        # 1. Top HUD Header Bar
        overlay = frame.copy()
        header_height = 56
        cv2.rectangle(overlay, (0, 0), (w, header_height), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(frame, "QUARREL AI", (14, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, self.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"SOURCE: {source_name[:18]} | FPS: {fps:.1f}", (14, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_GRAY, 1, cv2.LINE_AA)

        # State Badge Color
        if state == "QUARREL DETECTED" or quarrel_confirmed:
            badge_color = self.COLOR_RED
            badge_text = "🚨 QUARREL DETECTED"
        elif state == "CONFRONTATION":
            badge_color = self.COLOR_AMBER
            badge_text = "⚠️ CONFRONTATION"
        elif state == "ATTENTION":
            badge_color = (0, 215, 255)
            badge_text = "⚡ ATTENTION"
        else:
            badge_color = self.COLOR_GREEN
            badge_text = "✓ NORMAL"

        # Draw State Badge
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        badge_x = int(w / 2 - tw / 2)
        cv2.rectangle(frame, (badge_x - 10, 12), (badge_x + tw + 10, 44), badge_color, -1)
        cv2.putText(frame, badge_text, (badge_x, 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0) if badge_color != self.COLOR_RED else self.COLOR_WHITE, 1, cv2.LINE_AA)

        # Audio status indicator
        is_shouting, audio_score, audio_rms, audio_avail = audio_status
        audio_x = w - 160
        if audio_avail:
            mic_text = "MIC: SHOUTING" if is_shouting else "MIC: MONITORING"
            mic_color = self.COLOR_RED if is_shouting else self.COLOR_GREEN
            cv2.circle(frame, (audio_x - 10, 28), 5, mic_color, -1)
            cv2.putText(frame, mic_text, (audio_x, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # 2. Risk Meter Bar (under header)
        meter_y = 56
        meter_h = 7
        cv2.rectangle(frame, (0, meter_y), (w, meter_y + meter_h), (40, 40, 40), -1)
        fill_w = int(w * risk_score)
        fill_color = self.COLOR_GREEN if risk_score < 0.40 else (self.COLOR_AMBER if risk_score < 0.70 else self.COLOR_RED)
        if fill_w > 0:
            cv2.rectangle(frame, (0, meter_y), (fill_w, meter_y + meter_h), fill_color, -1)

        # 3. Dynamic Confrontation Line between subjects
        if len(tracked_faces) >= 2:
            for i in range(len(tracked_faces)):
                for j in range(i + 1, len(tracked_faces)):
                    c1 = (int(tracked_faces[i]['center'][0] * w), int(tracked_faces[i]['center'][1] * h))
                    c2 = (int(tracked_faces[j]['center'][0] * w), int(tracked_faces[j]['center'][1] * h))
                    
                    dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) / w
                    line_color = self.COLOR_RED if dist < 0.25 else (self.COLOR_AMBER if dist < 0.40 else self.COLOR_GREEN)
                    cv2.line(frame, c1, c2, line_color, 2, cv2.LINE_AA)
                    
                    mid_pt = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                    cv2.putText(frame, f"PROX: {dist:.2f}", (mid_pt[0] - 30, mid_pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # 4. Face Bounding Boxes & Identifiers
        for face in tracked_faces:
            bbox = face['bbox']
            x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
            x2, y2 = int(bbox[2] * w), int(bbox[3] * h)

            # Pad bounding box slightly
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.20)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            box_color = self.COLOR_RED if risk_score > 0.65 else self.COLOR_CYAN

            # Corner brackets
            corner_len = min(18, (x2 - x1) // 4)
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 2)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 2)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 2)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 2)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 2)

            # Person tag
            person_tag = f"Subject #{face['id']}"
            cv2.putText(frame, person_tag, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            # Key facial dots (nose, lips)
            lms = face['landmarks']
            if len(lms) > 14:
                # Nose tip
                nx, ny = int(lms[1][0] * w), int(lms[1][1] * h)
                cv2.circle(frame, (nx, ny), 2, (0, 255, 255), -1)
                # Upper & lower lip
                cv2.circle(frame, (int(lms[13][0] * w), int(lms[13][1] * h)), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lms[14][0] * w), int(lms[14][1] * h)), 2, (0, 0, 255), -1)

        # 5. Bottom Info Bar
        info_y = h - 14
        details = eval_result.get('details', {})
        f_agg = details.get('face_aggression', 0.0)
        c_mot = details.get('corridor_motion', 0.0)
        risk_pct = risk_score * 100

        info_str = f"RISK: {risk_pct:4.1f}% | FACIAL AGGRESSION: {f_agg:.2f} | MOTION ENERGY: {c_mot:.2f} | SUBJECTS: {len(tracked_faces)}"
        cv2.putText(frame, info_str, (12, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # 6. Flashing Red Border on Confirmed Quarrel
        if (state == "QUARREL DETECTED" or quarrel_confirmed) and (self.flash_counter % 8 < 4):
            border_thick = 8
            cv2.rectangle(frame, (0, 0), (w, h), self.COLOR_RED, border_thick)

        return frame
