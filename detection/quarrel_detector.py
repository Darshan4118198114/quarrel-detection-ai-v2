def check_interaction(centers, prev_centers):
    if len(centers) < 2 or prev_centers is None:
        return False, 0

    interaction_score = 0

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1 = centers[i]
            x2, y2 = centers[j]

            px1, py1 = prev_centers[i]
            px2, py2 = prev_centers[j]

            dist_now = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
            dist_prev = ((px1 - px2)**2 + (py1 - py2)**2) ** 0.5

            # moving closer
            if dist_prev - dist_now > 10:
                interaction_score += 1

    return interaction_score > 0, interaction_score