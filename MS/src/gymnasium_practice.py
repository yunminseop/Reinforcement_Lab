import gymnasium

env = gymnasium.make("Taxi-v3")
observation = env.reset()

for _ in range(1000):
    env.render() # render = 화면에 출력하는 용도
    action = env.action_space.sample() # 환경에 따라 적절한 액션을 설정
    observation, reward, terminated, truncated, info = env.step(action) 
    """ observation: 액션의 결과, reward: 보상, terminated: 에피소드 종료, truncated: 시간 초과 등으로 강제 종료, info: 추가 정보"""

    