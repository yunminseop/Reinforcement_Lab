def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']
    is_offtrack = params['is_offtrack']
    

    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    

    reward = 0

    # if agent is on the track
    if not is_offtrack:
        reward += 0.1

    # if steering_angle exceed the threshold,
    if abs(steering_angle) >= 15:
        curve = True
        if steering_angle < 0:
            turn_right = True
        else:
            turn_right = False
    else:
        curve = False

    if not curve: # drive straight
        if distance_from_center <= marker_1:
            reward += 1.0
        elif distance_from_center <= marker_2:
            reward += 0.3
        elif distance_from_center <= marker_3:
            reward += 0.1
        else:
            reward += 1e-3

    else: # drive curve
        if turn_right:
            if not is_left_of_center:
                if marker_2 <= distance_from_center < marker_3:
                    reward += 1.0
                else:
                    reward += 0.1
            else:
                reward += 1e-3
        else:
            if is_left_of_center:
                if marker_2 <= distance_from_center < marker_3:
                    reward += 1.0
                else:
                    reward += 0.1
            else:
                reward += 1e-3
            

            
    
    return float(reward)