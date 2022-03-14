# Lecture1
## Part1
### Introduction
 바구니에 물체를 집어 올리는 pick and place 로봇을 상상해보자. 로봇의 카메라로부터 이미지를 얻어 물체를 집어올리기 위해 gripper를 배치하는 시스템을 디자인해야한다. 디자인한 시스템에 물체가 담긴 바구니 이미지를 입력하면, gripper가 배치될 (x,y,z)좌표를 출력해야한다고 할 때, 여러 방법으로 문제를 접근해볼 수 있다.  예를 들어,
 1) Robotic system적으로 문제를 해결한다. 

    robotic system 지식을 활용하여 카메라와 gripper의 좌표간의 관계를 계산하고, 3D reconstruction algorithm을 통해 물체의 geometry와 position을 계산한다. 이를 통해 gripper를 어떻게 움직여 물체를 집어낼지 찾는다.
   다음 방법이 잘 작동하는 경우도 있으나 물체에 따라 정보가 더 요구된다. 예를 들어, 크고 무거운 물체에 대해선 무게 중심을 잘 알아야한다. 또한 연성인 물체에 대해선 딱딱한 물체에 적용하였던 사전 지식과는 다르게 중심을 꼬집어 잡는 방법이 효과적이다.

 2) Machine learning problem으로 접근한다. 

    다만, 기존의 지도 학습 방법으로 접근하기 어렵다. 이미지 라벨링과 물체 별 xyz position이 필요하고, 로봇의 입장에 gripper를 어디에 위치시켜야하는지 직관적으로 찾기 어렵기 때문이다.    

제시한 두가지 방법은 각각 한계가 존재하였다. 따라서 해당 강의의 목표는 강화학습을 통해 문제 해결을 자동화하는 것이다. 고난이도 레벨에서는 강화학습을 통해 실험 데이터에 대한 결과를 예측하도록 한다. 이때 데이터는 annotation이 포함된 ground-truth 데이터일 필요도 없고, 성공적인 행동으로 부터 나온 데이터일 필요도 없다. pick and place 로봇이 여러 물체를 집어 보면서 성공 경험이나 실패 경험의 데이터를 grasping system에 제공한다. 사람이나 이미지 데이터를 활용하는 대신 로봇들이 자동으로 데이터를 축적한다.  이후, gripper의 (x,y,z) 좌표에 대해 {성공 or 실패}로 라벨링된 데이터는 강화학습 알고리즘에 사용되어 물체를 더 잘 집을 수 있는 policy를 배우게 된다. 더 나아가, 최종 policy를 실제 문제에 적용해보면서 추가적인 데이터를 얻고, 이를 활용하여 policy를 더욱 개선시킬 수 있다. 
### - 강화학습이란 무엇인가.
강화학습은 2가지로 설명할 수 있다. 
1) learning에 기반한 의사결정 문제를 위한 수학적인 형식 (mathematical formalism) **@질문**
2) 제어기에 의존하거나 사람이 직접 시스템을 디자인하지 않고, 오로지 경험을 통해 의사결정 및 제어를 위한 접근 방법

### - 다른 Machine learning 주제와 강화학습이 다른점은 무엇인가.
기존 **지도 학습**의 목표는 input으로 주어진 (X,Y)에 대해서 x로 부터 y를 예측할 수 있는 함수를 배우는 것이다. 여기서 가정하고 있는 것은 
1) 데이터가 독립동일분포(independent identically distributed, iid)라는 것이다. 이는 특정 x로 부터 결과 y가 도출되었을 때 다른 x의 영향은 받지 않는 다는 특성이다. 
2) 또한, 학습 시에 ground truth ouput을 알고 있다는 가정이 존재한다.  

반면, **강화학습**은 
1) iid data를 가정하고 있지 않으며 이전의 결과가 미래의 input에 영향을 준다. 현재 어떤 action을 취하는지는 다음 observation에 영향을 미친다. 
2) 또한 ground truth answer을 알지 못한다. 성공 혹은 실패에 대한 결과만 알 수 있으며, 더 정확히는 해당 action에 대한 reward 정보만 받아온다. 

강화학습은 agent와 환경이 서로 상호작용하면서 의사 결정 시스템을 모델링하는 것이다. agent는 행동(action)을 결정하며, 환경은 그 행동에 대한 결과로 agent에게 observation(state)과 보상(reward)를 돌려준다. 한 에피소드 당 다음 상호작용은 여러번 반복되는데 고정된 숫자만큼 반복할 수도 있고(finite horizon), 그 반대로 무한정 반복할 수도 있다(infinite horizon). 
강화학습을 실생활에 대입해보면, 개를 훈련한다고 할 때 action은 개의 근육 수축, observation은 시각, 후각 등이 될 수 있으며, reward는 훈련에 맞게 행동했을 시 보상으로 주어지는 음식이다. 로봇을 훈련한다고하면, action은 모터의 전류나 회전력, observation은 카메라 이미지와 같이 부착된 센서로 부터 얻은 데이터, reward는 task 별 성공 척도(예를 들어,달리기task에서는 로봇의 속도)이다. 이 뿐만 아니라, 제고 관리등에도 활용될 수 있다. 이때 action은 어떤 물건을 구매할 것인지, observation은 현시점의 제고 상태, reward는 수익이 될 수 있다. 더 나아가, 망치질과 같은 복잡한 물리적인 문제, Atari breakout과 같은 게임에도 적용할 수 있다. 앞서 언급되었던 pick and place 로봇의 경우, 다양한 물체 집어올리기를 경험한 다음, 뭉쳐있는 물체들 중에서 하나씩 집어올리는 것과 같이 더 심화된 문제를 해결하는데 활용될 수 있다. 다양한 물체를 집어올릴 수 있도록 policy를 훈련할 수도 있는데, 로봇이 물체를 집어올리기 어렵도록 교란이 있을때도 다시 재시도 할 수 있도록 generalization을 학습해야한다. 

### - 강화학습은 게임이나 로봇만을 위한 게 아니다. 
재고관리나 Cathy Wu의 연구인 교통 시스템에도 적용할 수 있다.
예를 들어, 타원형 도로에서 강화학습을 적용한 자율 주행 차량이 자신의 속도를 동적으로 조절하면서 모든 차량이 적정한 속도를 유지할 수 있도록 할 수 있다. 초기에는 여러 차량에의 움직임에 의해 traffic jam이 발생하나, 시간이 지날 수록 자율 주행 차량이 차량 흐름을 조절할 수 있는 policy를 배우며 교통 상황이 원만해진다. 'eight situation(8자 형태의 도로)'에서도 마찬가지로 강화학습 자율 주행차량을 통해 모든 차량이 교차로에서 기달리지 않고 원만히 주행할 수 있도록 차량 흐름을 조절할 수 있다. 

## Part2
### - 왜 **'deep'** 강화학습이 필요한가.
### - 어떻게 똑똑한 기계(intelligent machine)를 개발할 수 있는가.
똑똑한 기계는 예측 불가능한 실제 상황에도 적응할 수 있는 능력이 있어야한다. 예를 들어, 자동 유조선을 개발할 때, 바다를 가로질러 지구 반대편으로 이동하기 위해 사람이 직접 운전하는 것은 어려울 수 있지만 gps와 motion planning 기술의 조합과 함께라면 이동에는 문제가 없을 것이다. 그럼에도 불구하고, 여전히 유조선에 기술자가 탑승하는 이유는 문제 상황이 발생했을 때 이를 해결하기 위해선 기술자가 필요하기 때문이다.  
  
### - Deep learning은 구조화되지 않은 환경에 효과적이다.
구조화되있지 않으며 예측하기 어려운 현실 세계의 문제를 다루기 위해선 딥러닝이 필요하다. 딥러닝에서는 deep neural network와 같이 매우 크고 초과 매개변수화(over-parameterized)된 모델을 통해 input이 ouput에 대응될 수 있도록 훈련시킨다. 예를 들어, 이미지내 물체를 인식하고자 할때, 엄청난 양의 라벨링된 이미지를 모으고 지도 학습방법을 사용하여 input을 예측할 수 있도록 훈련시킨다. 하지만 딥러닝은 본질적으로 어떤 알고리즘을 선택하냐보다 얼마나 크고 초과 매개변수화된 모델을 선택하냐의 문제이다. 딥러닝을 통해 이미지 분류, 번역, 사람의 말을 인식하는 문제까지 잘 해결하였으며, 다음 문제는 경험하지 않은 상황에도 잘 적응할 수 있는 일반화 된 모델이 필요하며, 예측 불허한 특이 상황이 발생할 수도 있다는 맥락에서 모두 open-world세팅으로 구분된다. 

### - 강화학습은 행동에 공식(formalism)을 제공한다.
강화학습은 순차적인 의사 결정 문제에 대해 수학적인 방법을 제공해준다. 강화학습에서 agent는 환경과 상호작용하며 observation과 reward를 얻는데, 강화학습에 deep neural network와 결합하여 특이하고 예측 불가한 상황에도 적응할 수 있도록 한다. 기술 초기의 성공적인 예시 중 하나로 90년대 board game, backgammon이 있다. 시스템 TD gammon은 전문가만큼으로 backgammon 게임을 수행한다. 또한 2016년 세계적인 바둑기사 이세돌을 이긴 Alphago도 TD gammon와 많은 공통점이 있다. Deep RL은 robotic locomotion부터 robotic manipulation, video game등에 활용되고 있다.

### - Deep RL은 무엇이고, 무엇을 알아야 하는가.
Deep RL의 중요성을 알기 위해서는 우선 computer vision 분야를 통해 DNN의 위대함을 살펴보자. computer vision 초반에는 pixel 이미지에 대해 HOG방법등을 통해 직접 low-level visual feature을 뽑아내고, DPM등을 통해서 middle-level feature을 뽑아낸 다음, 최종적으로 SVM과 같은 선형 분류기를 사용하여 이미지 내 객체를 분류하였다. 하지만 DNN의 경우에는 feature을 직접 뽑아낼 필요가 없어졌고, DNN을 통해 end-to-end학습을 수행하게 되었다. 이는 인간의 노고를 줄여줄 뿐만 아니라, features가 자동적으로 해결하고자 하는 문제에 맞게 뽑아진다는 데에 의의가 있다. 단순히 일반적인 feature을 추출하는 것이 아니라, 제규어와 호랑이 구분하기와 같이 실제 문제를 푸는데 필요한 feature을 추출할 수 있다. 다시 강화학습의 문제로 돌아와 backgammon게임을 생각해보자. 전통적인 강화학습의 경우에는 게임에서 action을 정할 때 사용할 중요 feature들을 추출해내야했다. game에서 중요시 된다고 생각하는 요소뿐만 아니라 정책이나 가치 함수 등 강화학습 구현 시 필요한 feature들도 필요하다. 따라서 좋은 feature을 추출해내기 위해선 backgammon게임 뿐만아니라, 강화학습에 대한 충분한 지식을 갖춰야한다. 이러한 이유 때문에 한동안 강화학습을 복잡한 문제에 적용하는것이 어려웠다. 하지만 computer vision과 마찬가지로 dnn을 강화학습에 적용하므로써 수동으로 feature추출을 해야했던 문제에서 벗어날 수 있었다. dnn에 의해 추출된 feature를 사용하여 end-to-end학습이 가능해졌다. 
평균적으로 우리가 다룰 폭넓은 문제에 대해 강화학습의 문제는 computer vision보다  feature 추출에 필요한 우리의 직관이 상당히 낮으며 이러한 이유 때문에 deep RL방법이 강화학습 알고리즘의 능력에 큰 영향을 미친다. 

### - 순차적 의사결정 문제에 end-to-end learning의 의미는 무엇인가.
intended learning 이 없다는 것의 의미는 recognition part와 control part를 분리해서 생각해야한다는 것이다. 우선적으로 이미지에서 재규어인지 사자인지 등을 파악해야하고, 그 다음 인식한 결과에 따라 어떤 행동을 취할 지 결정하는 파이프라인이 필요하다. 따라서 높은 인식률을 가지도록 perception 시스템을 훈련시켜야하고, 올바른 action을 취하기 위해 control 시스템도 훈련시켜야한다. 두 시스템이 분리되기 때문에 perception시스템은 어떤 detection이 중요한지에 대한 control system의 요구를 알지 못한다. 
하지만 intense system 은 sensory motor loop를 close 하지만, end-to-end로 perception과 control에 대한 전체 시스템을 학습한다. task의 최종 시각적인 데이터와 행동 features를 전부 얻는다.    
예를들어, robotic control 문제에 대해서 살펴보자.
전통적인 robotics 파이프라인은 observation을 얻고 물체의 위치 등을 추정하고, 모델링과 예측을 통해서 어떻게 행동할 지 계산한다. 이후 그 정보에 기반하여 planning을 하고, low-level control을 진행한다. 중요하게도, 각 단계에서 실수를 하여 에러를 발생시킬 수 있다.  intent 접근은 다음 한계를 극복할 수 있는데, 각 단계는 다음 단계의 요구를 알 수 있기 때문이다. 따라서 CNN을 사용한 강화학습은 perception 과 action을 동시에 수행한다. 로봇의 카메라로부터의 이미지는 network의 인풋으로 들어가고, stage끝단에서의 output은 로봇의 actuator에 들어간다. 다음과 같은 sensory motor loop은 시각적 요소와 행동적 요소가 결합한 작은 뇌의 축소판이다. 그리고 전부 task의 끝단에서 end-to-end 학습된다. CNN layer은 매우 작고 고도화된 visual cortex로 볼 수 있고, fully connected layer는 motor cortex로 볼 수 있다. 그들은 마지막에 end-to-end로 훈련되기 때문에 그것들의 능력안에서 최적으로 작동하려할 것이다. 이를 통해서 로봇은 경험으로 부터 배울 수 있고 neural network의 weight들을 훈련시킬 것이다.               
여러 예시들에대해 강화학습에 DNN을 결합한다고 생각해보면, 강화학습은 전체 AI 문제를 커버할 수 있다. 지도학습은 input과 ouput supervision이 필요하지만, 강화학습은 그러한 supervision없이 reward 피드백 만으로도 최적의 행동을 목표로 한다. 그리고 DNN은 강화학습 알고리즘이 복잡한 문제를 end-to-end로 풀 수 있게 한다. 그래서 강화학습은 mathematical formalism 을 제공하고, DNN은 실세계 문제로 확장될 수 있도록 표현방법(representation)을 제시한다. 

### - 왜 우리는 지금 Deep RL을 배워야 하는가.
1) deep learning의 발전 : DNN을 구성하는 방법의 발전(dnn 표현법)
2) 강화학습의 발전 : dnn의 표현법을 사용할 수 있게됨.
3) 계산 능력의 발전 : dnn + 강화학습을 합칠 수 있게됨.

Deep RL의 역사는 이제막 시작된 것은 아니다. 예를 들어, 1980년대 'Neural Networks for Control'이란 책에서와 같이 강화학습이 Neural Network와 함꼐 사용될 수 있다는 얘기등이 언급되었다. 