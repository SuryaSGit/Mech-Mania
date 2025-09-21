from . import *

def get_strategy(team: int):
    """This function tells the engine what strategy you want your bot to use"""
    
    # team == 0 means I am on the left
    # team == 1 means I am on the right
    
    if team == 0:
        print("Hello! I am team A (on the left)")
        return Strategy(goalee_formation, ball_chase)
    else:
        print("Hello! I am team B (on the right)")
        return Strategy(goalee_formation, ball_chase)

    # NOTE when actually submitting your bot, you probably want to have the SAME strategy for both
    # sides.

def goalee_formation(score: Score) -> List[Vec2]:
    """The engine will call this function every time the field is reset:
    either after a goal, if the ball has not moved for too long, or right before endgame"""
    
    config = get_config()
    field = config.field.bottom_right()
    
    return [
        Vec2(field.x * 0.1, field.y * 0.5),
        Vec2(field.x * 0.5, field.y * 0.5),
        Vec2(field.x * 0.5, field.y * 0.9),
        Vec2(field.x * 0.4, field.y * 0.3),
    ]

def GetGoalieAction(game: GameState) -> PlayerAction:
    """
    Goalkeeper positioning on the edges of the penalty box rectangle
    """
    config = get_config()
    goalie = game.players[0]  # Assuming goalie is player 0
    ball_pos = game.ball.pos
    goal_center = config.field.goal_self()
    field = config.field.bottom_right()
    
    # Penalty box dimensions
    penalty_box_width = config.goal.penalty_box_width  # Distance from goal line
    penalty_box_height = config.goal.penalty_box_height  # Height of the box
    
    # Define penalty box rectangle edges
    penalty_left_x = goal_center.x
    penalty_right_x = goal_center.x + penalty_box_width
    penalty_top_y = goal_center.y - penalty_box_height / 2
    penalty_bottom_y = goal_center.y + penalty_box_height / 2
    
    # Calculate optimal position on penalty box edge based on ball position
    ball_to_goal = goal_center - ball_pos
    
    # Default to right edge positioning (most common)
    if ball_pos.x <= goal_center.x:
        # Ball behind goal - stay on right edge, centered
        target_pos = Vec2(penalty_right_x, goal_center.y)
    else:
        # Ball in front - find best edge position
        if abs(ball_to_goal.x) < 0.001:  # Ball directly in line with goal
            target_y = goal_center.y
            target_pos = Vec2(penalty_right_x, target_y)
        else:
            # Calculate where ball-to-goal line intersects penalty box right edge
            t = (penalty_right_x - goal_center.x) / ball_to_goal.x
            target_y = goal_center.y + t * ball_to_goal.y
            
            # Check if intersection is within penalty box height
            if penalty_top_y <= target_y <= penalty_bottom_y:
                # Position on right edge
                target_pos = Vec2(penalty_right_x, target_y)
            elif target_y < penalty_top_y:
                # Position on top edge
                if abs(ball_to_goal.y) > 0.001:
                    t_top = (penalty_top_y - goal_center.y) / ball_to_goal.y
                    target_x = goal_center.x + t_top * ball_to_goal.x
                    target_x = max(penalty_left_x, min(penalty_right_x, target_x))
                    target_pos = Vec2(target_x, penalty_top_y)
                else:
                    target_pos = Vec2(penalty_right_x, penalty_top_y)
            else:
                # Position on bottom edge
                if abs(ball_to_goal.y) > 0.001:
                    t_bottom = (penalty_bottom_y - goal_center.y) / ball_to_goal.y
                    target_x = goal_center.x + t_bottom * ball_to_goal.x
                    target_x = max(penalty_left_x, min(penalty_right_x, target_x))
                    target_pos = Vec2(target_x, penalty_bottom_y)
                else:
                    target_pos = Vec2(penalty_right_x, penalty_bottom_y)
    
    # Calculate movement toward target position
    movement = target_pos - goalie.pos
    
    # If ball is very close and goalie can reach it, go for the ball
    ball_distance = (ball_pos - goalie.pos).norm()
        
    # If we have the ball, try simple actions
    if ball_distance < config.player.pickup_radius:
        return PlayerAction(Vec2(0, 0), config.field.goal_other() - goalie.pos)
    
    # Limit movement speed for controlled positioning
    max_movement = 1
    if movement.norm() > max_movement:
        movement = movement.normalize() * max_movement
    
    return PlayerAction(movement, None)

def calculate_wall_shot_to_goal(player_pos, goal_pos, field):
    """Calculate where to shoot on wall to bounce into goal"""
    
    # Try top wall bounce
    if player_pos.y > field.y * 0.3:  # Player not too close to top wall
        # Calculate reflection point on top wall (y = 0)
        wall_hit_x = (goal_pos.x * player_pos.y + player_pos.x * goal_pos.y) / (player_pos.y + goal_pos.y)
        
        if 0 <= wall_hit_x <= field.x:
            bounce_point = Vec2(wall_hit_x, 0)
            return (bounce_point - player_pos).normalize()
    
    # Try bottom wall bounce
    if player_pos.y < field.y * 0.7:  # Player not too close to bottom wall
        denominator = (field.y - player_pos.y) + (goal_pos.y - field.y)
        if abs(denominator) > 0.001:
            wall_hit_x = (goal_pos.x * (field.y - player_pos.y) + player_pos.x * (field.y - goal_pos.y)) / denominator
            
            if 0 <= wall_hit_x <= field.x:
                bounce_point = Vec2(wall_hit_x, field.y)
                return (bounce_point - player_pos).normalize()
    
    # Fallback: direct shot at goal
    return (goal_pos - player_pos).normalize()

def is_shot_blocked(start_pos, target_pos, enemies, config):
    """Simple function to check if anyone is blocking a shot path"""
    shot_vector = target_pos - start_pos
    shot_distance = shot_vector.norm()
    
    if shot_distance == 0:
        return False
    
    shot_direction = shot_vector.normalize()
    
    for enemy in enemies:
        # Vector from shot start to enemy
        to_enemy = enemy.pos - start_pos
        
        # Project enemy onto shot line
        projection = to_enemy.dot(shot_direction)
        
        # Check if enemy is along the shot path (not behind shooter)
        if 0 < projection < shot_distance:
            # Distance from enemy to shot line (2D cross product: ax*by - ay*bx)
            distance_to_line = abs(to_enemy.x * shot_direction.y - to_enemy.y * shot_direction.x)
            
            # If enemy is close enough to block shot
            if distance_to_line < config.player.radius * 2:
                return True
    
    return False

def calculate_wall_shot_to_corner(player_pos, corner_pos, field, enemies=None, config=None):
    """Calculate where to shoot on wall to bounce into corner, checking for blocks"""
    
    # Try top wall bounce (y = 0)
    if player_pos.y > field.y * 0.3 and corner_pos.y != 0:
        wall_hit_x = (corner_pos.x * player_pos.y + player_pos.x * corner_pos.y) / (player_pos.y + corner_pos.y)
        if 0 <= wall_hit_x <= field.x:
            bounce_point = Vec2(wall_hit_x, 0)
            shot_direction = (bounce_point - player_pos).normalize()
            
            # Check if shot is blocked (if enemies provided)
            if enemies and config and is_shot_blocked(player_pos, bounce_point, enemies, config):
                pass  # Try next wall
            else:
                return shot_direction
    
    # Try bottom wall bounce (y = field.y)
    if player_pos.y < field.y * 0.7 and corner_pos.y != field.y:
        denominator = (field.y - player_pos.y) + (corner_pos.y - field.y)
        if abs(denominator) > 0.001:
            wall_hit_x = (corner_pos.x * (field.y - player_pos.y) + player_pos.x * (field.y - corner_pos.y)) / denominator
            if 0 <= wall_hit_x <= field.x:
                bounce_point = Vec2(wall_hit_x, field.y)
                shot_direction = (bounce_point - player_pos).normalize()
                
                # Check if shot is blocked
                if enemies and config and is_shot_blocked(player_pos, bounce_point, enemies, config):
                    pass  # Try next wall
                else:
                    return shot_direction
    
    # Try left wall bounce (x = 0)
    if player_pos.x > field.x * 0.3 and corner_pos.x != 0:
        wall_hit_y = (corner_pos.y * player_pos.x + player_pos.y * corner_pos.x) / (player_pos.x + corner_pos.x)
        if 0 <= wall_hit_y <= field.y:
            bounce_point = Vec2(0, wall_hit_y)
            shot_direction = (bounce_point - player_pos).normalize()
            
            # Check if shot is blocked
            if enemies and config and is_shot_blocked(player_pos, bounce_point, enemies, config):
                pass  # Try next wall
            else:
                return shot_direction
    
    # Try right wall bounce (x = field.x)  
    if player_pos.x < field.x * 0.7 and corner_pos.x != field.x:
        denominator = (field.x - player_pos.x) + (corner_pos.x - field.x)
        if abs(denominator) > 0.001:
            wall_hit_y = (corner_pos.y * (field.x - player_pos.x) + player_pos.y * (field.x - corner_pos.x)) / denominator
            if 0 <= wall_hit_y <= field.y:
                bounce_point = Vec2(field.x, wall_hit_y)
                shot_direction = (bounce_point - player_pos).normalize()
                
                # Check if shot is blocked
                if enemies and config and is_shot_blocked(player_pos, bounce_point, enemies, config):
                    pass  # Try next wall
                else:
                    return shot_direction
    
    # Fallback: shoot toward corner directly (only if not blocked)
    direct_shot = (corner_pos - player_pos).normalize()
    if enemies and config and is_shot_blocked(player_pos, corner_pos, enemies, config):
        return None  # All shots blocked
    
    return direct_shot

def ball_chase(game: GameState) -> List[PlayerAction]:
    """Enhanced strategy with intelligent support players"""
    
    config = get_config()
    actions = []
    
    # Goalkeeper action
    actions.append(GetGoalieAction(game))
    
    # Main striker (player 1)
    player_pos = game.players[1].pos
    field = config.field.bottom_right()
    enemy_goal = config.field.goal_other()
    goal_height = config.goal.normal_height
    ball_pos = game.ball.pos
    
    # Check who has the ball
    ball_holder = None
    ball_distance_threshold = config.player.pickup_radius * 1.2
    
    # Check our team for ball possession
    for i in range(NUM_PLAYERS // 2):
        if (game.players[i].pos - ball_pos).norm() <= ball_distance_threshold:
            ball_holder = ('our_team', i)
            break
    
    # Check enemy team for ball possession
    if ball_holder is None:
        for i in range(NUM_PLAYERS // 2, NUM_PLAYERS):
            if (game.players[i].pos - ball_pos).norm() <= ball_distance_threshold:
                ball_holder = ('enemy_team', i)
                break
    
    # Main striker behavior (player 1)
    goal_corners = [
        Vec2(enemy_goal.x, enemy_goal.y - goal_height/2),  # Top goal corner
        Vec2(enemy_goal.x, enemy_goal.y + goal_height/2),  # Bottom goal corner
    ]
    target_corner = Vec2(enemy_goal.x, enemy_goal.y - goal_height/2)
    
    enemies = [game.players[i] for i in range(NUM_PLAYERS // 2, NUM_PLAYERS)]
    wall_shot = calculate_wall_shot_to_corner(player_pos, target_corner, field, enemies, config)
    
    if wall_shot is not None:
        actions.append(PlayerAction(ball_pos - player_pos, wall_shot))
    else:
        # Fallback to direct shot
        direct_shot = (target_corner - player_pos).normalize()
        actions.append(PlayerAction(ball_pos - player_pos, direct_shot))
    
    # Support players (2 and 3) behavior
    for support_idx in [2, 3]:
        if support_idx >= NUM_PLAYERS:
            actions.append(PlayerAction(Vec2(0, 0), None))
            continue
            
        support_pos = game.players[support_idx].pos
        
        if ball_holder is None:
            # Ball is free - run towards it
            to_ball = ball_pos - support_pos
            if to_ball.norm() > 0:
                movement = to_ball.normalize()
            else:
                movement = Vec2(0, 0)
            actions.append(PlayerAction(movement, None))
            
        elif ball_holder[0] == 'our_team':
            # Our team has the ball
            if ball_holder[1] == support_idx:  # This support player has the ball
                # 1. Check for open shot opportunities
                goal_corners = [
                    Vec2(enemy_goal.x, enemy_goal.y - goal_height/2),  # Top corner
                    Vec2(enemy_goal.x, enemy_goal.y + goal_height/2),  # Bottom corner
                ]
                
                has_open_shot = False
                shot_direction = None
                
                # Check direct shots to corners
                for corner in goal_corners:
                    if not is_shot_blocked(support_pos, corner, enemies, config):
                        has_open_shot = True
                        shot_direction = (corner - support_pos).normalize()
                        break
                
                # If no direct shot, try wall shots
                if not has_open_shot:
                    for corner in goal_corners:
                        wall_shot = calculate_wall_shot_to_corner(support_pos, corner, field, enemies, config)
                        if wall_shot is not None:
                            has_open_shot = True
                            shot_direction = wall_shot
                            break
                
                if has_open_shot:
                    # Take the shot
                    actions.append(PlayerAction(Vec2(0, 0), shot_direction))
                else:
                    # 2. No open shot - count nearby enemies
                    nearby_enemies = []
                    enemy_threshold = config.player.radius * 8  # Define "nearby"
                    
                    for enemy in enemies:
                        if (enemy.pos - support_pos).norm() <= enemy_threshold:
                            nearby_enemies.append(enemy)
                    
                    if len(nearby_enemies) >= 2:
                        # 3. Two or more enemies nearby - look for pass
                        best_pass_target = None
                        best_pass_score = -1
                        
                        # Check teammates for passing options
                        for teammate_idx in range(NUM_PLAYERS // 2):
                            if teammate_idx == support_idx:  # Don't pass to self
                                continue
                            
                            teammate_pos = game.players[teammate_idx].pos
                            
                            # Check if pass is clear
                            if not is_shot_blocked(support_pos, teammate_pos, enemies, config):
                                # Calculate pass score (closer to goal is better, more open is better)
                                goal_distance = (enemy_goal - teammate_pos).norm()
                                goal_score = 1.0 - (goal_distance / field.norm())
                                
                                # Check how open teammate is
                                min_enemy_dist = min((enemy.pos - teammate_pos).norm() for enemy in enemies)
                                openness_score = min(1.0, min_enemy_dist / (config.player.radius * 6))
                                
                                total_score = goal_score * 0.6 + openness_score * 0.4
                                
                                if total_score > best_pass_score:
                                    best_pass_score = total_score
                                    best_pass_target = teammate_pos
                        
                        if best_pass_target is not None:
                            # Make the pass
                            pass_direction = (best_pass_target - support_pos).normalize()
                            actions.append(PlayerAction(Vec2(0, 0), pass_direction))
                        else:
                            # No good pass - dribble toward goal
                            dribble_direction = (enemy_goal - support_pos).normalize() * 0.8
                            actions.append(PlayerAction(dribble_direction, None))
                    else:
                        # 4. One or no enemies nearby - move up and try to score
                        # Move toward goal while looking for shooting opportunity
                        to_goal = (enemy_goal - support_pos).normalize()
                        
                        # Try to get to a better shooting position
                        ideal_shooting_x = field.x * 0.75
                        if support_pos.x < ideal_shooting_x:
                            # Move forward toward ideal shooting range
                            movement = to_goal * 0.8
                        else:
                            # Close to goal - look for angle
                            if support_pos.y < field.y * 0.4:
                                # Move toward center-top
                                movement = Vec2(to_goal.x * 0.5, 0.5)
                            elif support_pos.y > field.y * 0.6:
                                # Move toward center-bottom  
                                movement = Vec2(to_goal.x * 0.5, -0.5)
                            else:
                                # Move straight forward
                                movement = to_goal
                        
                        if movement.norm() > 1.0:
                            movement = movement.normalize()
                        actions.append(PlayerAction(movement, None))
                        
            elif ball_holder[1] == 1:  # Main striker (player 1) has ball
                # Get open and move down field to receive pass
                # Move towards enemy goal but spread out
                offset = 0.3 if support_idx == 2 else -0.3  # Spread vertically
                target_x = min(field.x * 0.8, support_pos.x + field.x * 0.2)  # Move forward
                target_y = field.y * 0.5 + field.y * offset  # Spread out
                target_pos = Vec2(target_x, target_y)
                
                # Avoid getting too close to enemies while getting open
                min_enemy_distance = float('inf')
                for enemy in enemies:
                    dist = (enemy.pos - support_pos).norm()
                    if dist < min_enemy_distance:
                        min_enemy_distance = dist
                
                # If too close to enemy, create space
                if min_enemy_distance < config.player.radius * 4:
                    # Find direction away from nearest enemy
                    nearest_enemy = min(enemies, key=lambda e: (e.pos - support_pos).norm())
                    away_from_enemy = (support_pos - nearest_enemy.pos).normalize()
                    target_pos = support_pos + away_from_enemy * config.player.radius * 2
                
                movement = target_pos - support_pos
                if movement.norm() > 1.0:
                    movement = movement.normalize()
                actions.append(PlayerAction(movement, None))
            else:
                # Another teammate has the ball - support positioning
                movement = (enemy_goal - support_pos).normalize() * 0.5
                actions.append(PlayerAction(movement, None))
                
        else:
            # Enemy has the ball - mark free enemies and intercept passes
            enemy_with_ball_idx = ball_holder[1]
            enemy_with_ball = game.players[enemy_with_ball_idx]
            
            # Find unmarked enemies (excluding the one with ball)
            unmarked_enemies = []
            for i in range(NUM_PLAYERS // 2, NUM_PLAYERS):
                if i != enemy_with_ball_idx:
                    enemy_pos = game.players[i].pos
                    # Check if this enemy is already being marked by other support player
                    other_support_idx = 3 if support_idx == 2 else 2
                    if other_support_idx < NUM_PLAYERS:
                        other_support_pos = game.players[other_support_idx].pos
                        distance_to_other_support = (enemy_pos - other_support_pos).norm()
                        if distance_to_other_support > config.player.radius * 3:  # Not being marked
                            unmarked_enemies.append(game.players[i])
            
            if unmarked_enemies:
                # Mark the nearest unmarked enemy
                target_enemy = min(unmarked_enemies, key=lambda e: (e.pos - support_pos).norm())
                
                # Position between the enemy and the ball to intercept passes
                ball_to_enemy = target_enemy.pos - ball_pos
                if ball_to_enemy.norm() > 0:
                    # Position slightly closer to ball than the enemy
                    intercept_pos = ball_pos + ball_to_enemy.normalize() * (ball_to_enemy.norm() * 0.7)
                    movement = intercept_pos - support_pos
                else:
                    # If enemy is on ball, just get close to mark them
                    movement = target_enemy.pos - support_pos
                    
                if movement.norm() > 1.0:
                    movement = movement.normalize()
                actions.append(PlayerAction(movement, None))
            else:
                # All enemies marked or none to mark - move toward ball
                to_ball = ball_pos - support_pos
                if to_ball.norm() > 0:
                    movement = to_ball.normalize() * 0.8
                else:
                    movement = Vec2(0, 0)
                actions.append(PlayerAction(movement, None))
    
    # Fill remaining players if any
    while len(actions) < NUM_PLAYERS:
        actions.append(PlayerAction(Vec2(0, 0), None))
    
    return actions


def do_nothing(game: GameState) -> List[PlayerAction]:
    """This strategy will do nothing :("""
    actions = []
    actions.append(GetGoalieAction(game))
    for _ in range(NUM_PLAYERS-1):
        actions.append(GetGoalieAction(game))
    return actions

def rush_formation(score: Score) -> List[Vec2]:
    """The engine will call this function every time the field is reset:
    either after a goal, if the ball has not moved for too long, or right before endgame"""
    
    config = get_config()
    field = config.field.bottom_right()
    
    # Optimized starting positions for new_strategy roles:
    return [
        Vec2(field.x * 0.3, field.y * 0.5),   # Player 0: Ball rusher - closer to center
        Vec2(field.x * 0.25, field.y * 0.85),  # Player 1: Back corner receiver - in back corner
        Vec2(field.x * 0.5, field.y * 0.9), # Player 2: Side field runner - side position
        Vec2(field.x * 0.5, field.y * 0.05),   # Player 3: Support - defensive position
    ]

def do_nothing(game: GameState) -> List[PlayerAction]:
    """This strategy will do nothing :("""
    
    return [
        PlayerAction(Vec2(0, 0), None) 
        for _ in range(NUM_PLAYERS)
    ]

def new_strategy(game: GameState) -> List[PlayerAction]:
    """Fast ball control strategy: Rush ball → Back corner → Side field → Goal shot"""
    
    config = get_config()
    actions = []
    
    # Get field dimensions and key positions
    field = config.field.bottom_right()
    enemy_goal = config.field.goal_other()
    ball_pos = game.ball.pos
    
    # Define player roles:
    # Player 0: Ball rusher (closest to ball at start)
    # Player 1: Back corner receiver
    # Player 2: Side field runner 
    # Player 3: Support/backup
    
    # Calculate distances to ball for all players
    distances_to_ball = [(i, (game.players[i].pos - ball_pos).norm()) for i in range(NUM_PLAYERS)]
    distances_to_ball.sort(key=lambda x: x[1])
    closest_to_ball = distances_to_ball[0][0]
    
    # Determine who has possession
    ball_holder = None
    for i in range(NUM_PLAYERS):
        if (game.players[i].pos - ball_pos).norm() <= config.player.pickup_radius:
            ball_holder = i
            break
    
    for i in range(NUM_PLAYERS):
        player_pos = game.players[i].pos
        movement = Vec2(0, 0)
        pass_target = None
        
        if i == 0:  # Ball rusher
            if ball_holder != 0:
                # Rush to the ball as fast as possible
                to_ball = ball_pos - player_pos
                
                movement = to_ball
            else:
                # Has the ball - pass to nearest teammate
                # nearest_bot = min(
                #     (j for j in range(NUM_PLAYERS) if j != 0),
                #     key=lambda j: (game.players[j].pos - player_pos).norm()
                # )
                # nearest_pos = game.players[nearest_bot].pos
                nearest_pos = game.players[1].pos  # Always pass to player 1 (back corner receiver)
                pass_direction = nearest_pos - player_pos
                if pass_direction.norm() > 0:
                    pass_target = pass_direction.normalize()
                else:
                    pass_target = Vec2(1.0, 0.0)  # Default forward pass
                
                # Move slightly toward goal while passing
                to_goal = enemy_goal - player_pos
                if to_goal.norm() > 0:
                    movement = to_goal * 0.3
        
        elif i == 1:  # Back corner receiver
            if ball_holder != 1:
                # Position in back corner and wait for pass
                back_corner_pos = Vec2(field.x * 0.25, field.y * 0.85)
                to_corner = back_corner_pos - player_pos
                if to_corner.norm() > 0:
                    movement = to_corner
            else:
                # Has the ball - pass to side field runner (player 2)
                side_field_pos = game.players[2].pos
                pass_direction = side_field_pos - player_pos
                if pass_direction.norm() > 0:
                    pass_target = pass_direction.normalize()  # Direction to player 2
                else:
                    pass_target = Vec2(1.0, 0.0)  # Default forward pass
                # Stay in position while passing
                movement = Vec2(0, 0)
        
        elif i == 2:  # Side field runner
            if ball_holder != 2:
                # Move up the side field toward goal
                side_target = Vec2(field.x * 0.8, field.y * 0.92)  # Side field position
                to_side = side_target - player_pos
                if to_side.norm() > 0:
                    movement = to_side
            else:
                # Has the ball - shoot at goal with maximum power
                to_goal = enemy_goal - player_pos
                if to_goal.norm() < 260.0:  # Only shoot if within 100 units of goal
                    pass_target = to_goal.normalize()
                else:
                    pass_target = Vec2(0.0, 0.0)  # Default forward
                    movement = Vec2(1.0, 0.0)  # Move toward goal while shooting
        
        else:  # Player 3: Support/backup
            if ball_holder != 3:
                # Move up the side field toward goal
                side_target = Vec2(field.x * 0.8, field.y * 0.1)  # Side field position
                to_side = side_target - player_pos
                if to_side.norm() > 0:
                    movement = to_side
            else:
                # Has the ball - shoot at goal with maximum power
                to_goal = enemy_goal - player_pos
                if to_goal.norm() > 0:
                    pass_target = to_goal.normalize()  # Shoot at goal
                else:
                    pass_target = Vec2(1.0, 0.0)  # Default forward
                movement = to_goal * 0.5  # Move toward goal while shooting
        
        # Ensure movement doesn't exceed max magnitude
        if movement.norm() > 1.0:
            movement = movement.normalize()
        
        actions.append(PlayerAction(movement, pass_target))
    
    return actions