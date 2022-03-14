## Supervised learning of Behaviors
### Terminology & notation
input 이미지가 있고 네트워크에 넣으면 label과 같은 output을 얻을 수 있다.
여기서 기존의 지도학습 용어부터 시작해서 강화학습용 문제로 전환하려고 한다. 따라서 input o를 observation으로 부르고, output a를 action이라고 칭할 것이다. 그리고 input과 output을 매핑시켜주는 가운데 모델 $\pi_\theta(a|o)$을 policy라고 부를 것이다. 여기서 $\theta$는 그 policy의 parameter이고 neural net에선 weight의 개념과 일치한다. 모델은 주어진 o에대한 a분포를 나타낸다. 강화학습에서는 순차적 의사 결정 문제를 다룰 것이기에 매 시각마다 input과 output이 발생한다. 그래서 발생하는 시각을 표시하기 위해 time t를 사용한다. $\{o_t -> \pi_\theta(a_t|o_t) -> a_t \}$ 보통 강화학습에서는 discrete time 문제를 다루기 떄문에 시간을 이산적인 step개념으로 가정한다. 또한 지도학습과 다르게 한 단계의 output은 그 다음의 input에 영향을 미친다. $a_t$는 $o_{t+1}$에 영향을 미친다. 그래서 만약에 해당 단계에서 호랑이를 인식하지 못했다면, 그 다음 단계에서는 호랑이가 나에게 좀 더 가까이 접근하는 불상사가 발생할 수 도 있다. 그래서 이러한 기본 개념을 제어를 위한 policy를 배우는데 확장해볼 수 있다. 따라서 label을 output으로 도출하는 것이 아니라, 조금더 행동에 가까운 output을 도출할 수 있다. 여기서 action은 이산적일 수 있으며 softmax distribution을 사용하여 호랑이를 만났을 때 취할 수 있는 action set중에서 하나를 고를 수 있다. 더불어 action은 continuous할 수 도 있는데 $\pi_\theta$가 continuous distribution의 parameter을 도출할 수 있다. 예를 들면, multivariate normal 또는 gaussian distribution 에서의 mean과 variance 말이다. 
  Observation $o_t$ 
  Action $a_t$
  Policy $\pi_\theta(a_t|o_t)$
  state $s_t$
  그리고 가끔 policy를 $\pi_\theta(a_t|s_t)$로 나타내기도 하는데, state가 보통 markovian state라고 가정하며, observation $o_t$ 은 그 state로부터의 결과이다. 그래서 보통은 $\pi_\theta(a_t|o_t)$ 이렇게 쓰지만 특별한 경우 $\pi_\theta(a_t|s_t)$로 나타낸다. 
state vs observation
치타가 가젤을 쫒는 상황을 가정하자. 그 해당 사진의 pixels로부터 치타와 가젤이 어디에 위치하고 있는지 알아내기 충분할 수도 있고 아닐 수도 있다. 하지만 이미지는 기저하고 있는 어떤 시스템의 물리(법칙)에 기반하고 있으며 그 시스템은 일종의 최소한의 표현이라고 하는 state를 가지고 있다. 따라서 이미지 이미지는 observation $o_t$ 이고, state은  시스템의 현재 구성 표현인데 여기서는 치타와 가젤의 위치, 그리고 들의 속력이다. 그래서 만약에 observation이 달라지면 full state가 추론되지 못하는 경우도 있다. 예를들어 자동차가 지나가서 치타가 가려진다면 그 이미지로 부터 모든 state를 알아낼 수 없다. 하지만 사실 치타는 그대로의 위치에 있을 것이기에 state는 변동이 없을 것이지만, 그 이미지로 부터 state를 찾기 어려워진것이다. state는 그 시스템의 실제 구성을 의미하며 observation은 state를 추론하기에 충분할 수도 있고 아닐 수도 있는 그 state로부터의 결과이다. 그래픽 모델의 용어를 사용하여 설명해볼 수 있는데      

<p align="center">
 <img src = "./img/1.png">
</p>
</img>

observation이 state로부터의 결과이므로 매 step마다 화살표 표시가 되어있고 이후, policy는 observation을 사용해서 aciton을 결정한다. 현재의 action은 다음 step의 state에 영향을 미친다.  
이 그래픽 모델을 보면 특정한 독립이 존재하는데 $\pi_\theta$가 정책으로 주어지고, transition probability로 $p(s_{t+1}|s_t, a_t)$ 이 존재할 때, $p(s_{t+1}|s_t, a_t)$ 이 $s_{t-1}$ 와 독립이라는 것이다. 따라서 이전 state를 몰라도 현재의 state를 알고만 있으면 다음 state로의 distribution을 계산할 수 있다. 즉, 미래는 현재가 주어졌다면 과거와 독립이다. 미래 state를 위해서 action을 결정할 때 어떻게 현재 상황까지 도달했는지 고려할 필요도, 기억할 필요도 없다는 것이다. 이것은 'Markov property'이며 강화학습, 순차적 의사 결정 문제에 정말 정말 중요한 개념이다. 이 속성이 없다면 전체의 history를 고려하지 않고선 최적의 policy를 계산할 수 없기 떄문이다. 하지만 만약에 policy가 state대신 observation에 기반한다면 observation도 다음과 같은 독립을 만족하는지 봐야한다. 현재의 observtaion이 미래의 state에 도달하기 위한 action을 결정하는데 충분한지 파악해야한다. 문제는 보통 observation은 Markov Property를 만족하지 않아 충분하지 않을 수도 있다는 것이다. 자동차가 있어서 이미지 속에 치타를 발견하지 못했다면 미래에 어떤것을 해야하는지 결정할 수 없을 것이다. 하지만 차가 지나가기 이전의 이미지를 보면서 치타가 어디있었는지 그 state를 기억할 수 있다. 따라서 과거의 observation은 현재 observation이 보지 못하는 의사 결정에 필요한 그 이상의 정보를 제공할 수 도 있다.       
강의에서 다룰 많은 강화학습 문제는 Markovian state를 요구한다. -> $\pi_\theta(a_t|s_t)$ 하지만 non-markovian observation 을 다루는 몇몇의 알고리즘이 존재하긴 하다.  

### Aside : notation
Richard Bellman에 의해 연구된 DP에서의 용어와 같게 state, action에 $s_t, a_t$를 사용할것인데 이것으 로보틱에서의 용어와 차이가 있다. 

### Imitation Learning
어떻게 policy를 학습할 수 있는지 살펴보자. 우선 데이터를 사용하는 지도학습의 이미지 분류기 모델과 같이 간단한 학습 방법부터 볼것이다. 다음의 예시는 Driving 이다. observation은 자동차의 카메라로부터의 이미지이고, action은 차량이 차선을 따라 달릴 수 있도록 핸들을 조절하는것이다. computer vision의 이미지 분류에서의 driving과 같이 접근하여 labeling 된 데이터를 사용해서 driving policy를 학습한다고 해보자. 따라서 사람으로 부터 이미지를 얻고 그에 해당하는 motor commands가 필요하다. 카메라로부터의 이미지와 사람이 핸들을 어떻게 조절하는지 기록하여 (image, command)의 거대한 훈련 데이터를 축적한다. 그다음 지도학습을 적용하여 image에 command가 대응될 수 있도록 훈련한다. 이것은 imitation learning이라고 불리고, 이것은 사람(demonstrator)의 행동을 복사한다는 점에서 behavioral cloning이라고도 불리는 특별한 케이스이다. 여기서 운전자는 컴퓨터보다 해당 작업을 잘 수행해내야하기에 전문가여야한다. 이것이 실제 작동하는지 물음을 던져보면, 다음 문제는 오랫동안 연구되어 왔다. deep imitation learning 이나 neural imitation learning 과 같은 연구는 1989년도 부터 연구되어왔다. 