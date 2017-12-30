# readme2tex-tests

# Лабораторная работа №1

Выполнил:
 - Рой Виктор
 - ННГУ, ф-т ИТММ, каф. МО ЭВМ, группа 381603м4

## Задание
 - Изучить метод обратного распространения ошибки;
 - Вывести математические формулы для вычисления градиентов функции ошибки по параметрам
нейронной сети и формул коррекции весов;
 - Спроектировать и разработать программную реализацию;
 - Подготовить отчет по проделанной работе. 
 
## Запуск решения
 - Установить python3 c библиотекой Numpy;
 - Запустить файл main.py с аргументами 
 	- '-h' помощь 
 	- '-n' количество тренировочных изображений
 	- '-t' количество тестовых изображений
 	- '-s' количество скрытых слоев
 	- '-l' скорость обучения

Пример запуска: python main.py -n 10000 -t 1000 -s 200 -l 0.005

## Метод обратного распространения ошибки
Функция ошибки кросс-энтропия:
<p align="center"><img alt="$$E=-\sum_{j=1}^{N_o}t_jlog(y_j) = -\sum_{j=1}^{N_o}t_jlog(f(\sum_{s=1}^{N_s}w_s_jf(\sum_{i=1}^{N_i}w_i_sx_i)))$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/b36cdd66cc930db82e0861639504c0eb.svg?invert_in_darkmode" align=middle width="413.50155pt" height="50.226165pt"/></p>

Функция активации на скрытом слое тангес:
<p align="center"><img alt="$$\phi(y_j)={{e^{2y_j}-1}\over{e^{2y_j}+1}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/ed95e6fc19764479cbcac06b6293beab.svg?invert_in_darkmode" align=middle width="117.602925pt" height="37.147275pt"/></p>

Функция активации на втором слое softmax:
<p align="center"><img alt="$$f(y_j)={{e^{y_j}}\over\sum_{j=1}^{n}e^{y_j}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/f8ce48e4708e476899413ca68cba9971.svg?invert_in_darkmode" align=middle width="126.424485pt" height="40.62036pt"/></p>

### Алгоритм
1. Инициализируем веса значениями из диапазона [0, 0.5]
2. Пока количество проходов < max_epoch делаем:

	Для всех картинок от 1 до number_train_images
	+ Подаем на вход x, суммируем cигналы на скрытом слое <img alt="$\text z_s=w_0_s+\sum_{i}^{N_i}w_i_s x_i$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/eeda333d528c18f1d8db2456882c31e6.svg?invert_in_darkmode" align=middle width="157.356705pt" height="32.25618pt"/>, применяем функцию активации <img alt="$\text v_s=\phi(z_s)$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/adc1e796afadfdd5b803be0df7e299b5.svg?invert_in_darkmode" align=middle width="74.87073pt" height="24.6576pt"/>

	+ Для каждого выходного нейрона суммируем взвешенные входящие сигналы <img alt="$\text y_j=w_0_j+\sum_{s}^{N_s}w_s_j f(z_s)$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/c1bc737f1671da53dbd9b061d8ed69a8.svg?invert_in_darkmode" align=middle width="184.330905pt" height="32.25618pt"/>, применяем функцию активации <img alt="$\text u_j=f(y_j)$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/40bfa776a17e8fdfdcb93afe79f0b7f4.svg?invert_in_darkmode" align=middle width="75.565215pt" height="24.6576pt"/>

	+ Считаем градиенты функции ошибки:

	<p align="center"><img alt="$${{\partial{E}}\over{\partial{w_s_j}}}={{\partial{E}}\over{\partial{y_j}}}{{\partial{y_j}}\over{\partial{w_s_j}}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/33a5033e2a26ab9a0fdd1e5448f360f8.svg?invert_in_darkmode" align=middle width="125.15745pt" height="38.51529pt"/></p>

	<p align="center"><img alt="$${{\partial{E}}\over{\partial{y_j}}}=u_j-t_j$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/f31d0c7bd21cc5253aa7b195bbef0523.svg?invert_in_darkmode" align=middle width="96.984855pt" height="38.51529pt"/></p>

	<p align="center"><img alt="$${{\partial{y_j}}\over{\partial{w_s_j}}}=v_s$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/73d12a65bbf307b427bcf40f4e138867.svg?invert_in_darkmode" align=middle width="73.424175pt" height="38.51529pt"/></p>

	<p align="center"><img alt="$${{\partial{E}}\over{\partial{w_s_j}}}=(u_j-t_j)v_s={\delta_j{v_s}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/902c32840fc99fdf2a8963a1f8f77d64.svg?invert_in_darkmode" align=middle width="186.6447pt" height="38.51529pt"/></p>

	<p align="center"><img alt="$${{\partial{E}}\over{\partial{w_i_s}}}={{\partial{E}}\over{\partial{z_s}}}{{\partial{z_s}}\over{\partial{w_i_s}}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/c8a6c938ccd9641d27cb2404ac4cf16c.svg?invert_in_darkmode" align=middle width="121.93533pt" height="36.27789pt"/></p>

	<p align="center"><img alt="$${{\partial{E}}\over{\partial{z_s}}}=\sum_{j=1}^{N_o}{{\partial{E}}\over{\partial{y_j}}}{{\partial{y_j}}\over{\partial{v_s}}}{{\partial{f}}\over{\partial{z_s}}}={{\partial{f}}\over{\partial{z_s}}}\sum_{j=1}^{N_o}{{\partial{E}}\over{\partial{y_j}}}{{\partial{y_j}}\over{\partial{v_s}}}={{\partial{f}}\over{\partial{z_s}}}(\sum_{j=1}^{N_o}{\delta_j^2w_s_j^2})$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/e5fe7703c9325ead4c97db212f417354.svg?invert_in_darkmode" align=middle width="427.5579pt" height="50.226165pt"/></p>

	<p align="center"><img alt="$${{\partial{E}}\over{\partial{w_i_s}}}={{\partial{f}}\over{\partial{z_s}}}(\sum_{j=1}^{N_o}{\delta_j^2w_s_j^2}){x_i}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/cb94e20f4bd01e87c2e1c95ef063383e.svg?invert_in_darkmode" align=middle width="180.84165pt" height="50.226165pt"/></p>

 	 В случае гиперболического тангенса: 

	<p align="center"><img alt="$${{\partial{f}}\over{\partial{z_s}}}=(1-v_s)(1+v_s)$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/a2d1f5c4c1cfd4eb7b679350d8aeb85d.svg?invert_in_darkmode" align=middle width="160.38198pt" height="36.27789pt"/></p>

 	 Тогда 

	<p align="center"><img alt="$${{\partial{E}}\over{\partial{w_s_j}}}={\delta_j{v_s}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/e047b5019a674502886d7230d6dfb0ad.svg?invert_in_darkmode" align=middle width="87.65658pt" height="38.51529pt"/></p>

	<p align="center"><img alt="$${{\partial{E}}\over{\partial{w_i_s}}}={{\delta_s}{x_i}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/abe9d2643906994afdf0355a0f9f34d3.svg?invert_in_darkmode" align=middle width="86.176365pt" height="36.27789pt"/></p>
	+ Корректируем веса в соответствии с градиентами функции ошибки:

	<p align="center"><img alt="$${w_i_s^{n+1}=w_i_s^{n}-\eta{{\partial{E}}\over{\partial{w_i_s}}}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/e3b96ae350a7aaac265e6ce1ad594b5a.svg?invert_in_darkmode" align=middle width="155.663805pt" height="36.27789pt"/></p>

	<p align="center"><img alt="$${w_s_j^{n+1}=w_s_j^{n}-\eta{{\partial{E}}\over{\partial{w_s_j}}}}$$" src="https://rawgit.com/leegao/readme2tex-tests/svgs/svgs/b1af6355d19d273f5321c89b4e081f8d.svg?invert_in_darkmode" align=middle width="160.22424pt" height="38.51529pt"/></p>

## Реализация 
Программа написана на Python 3.6. В рамках лабораторной работы был создан класс NeuraNetwork.py, в котором реализованы следущие методы:

 - initializeWeights() - инициализация весов случайными значениями из диапазона [0, 0.5];
 - train(self, x_values, t_values, maxEpochs, learnRate, crossError) - обучение сети с помощью метода обратного распространения ошибки;
 - computeOutputs(self, xValues) - расчет значений на выходе сети;
 - computeGradient(self, t_values, oGrads, hGrads) - расчет градиентов для обновления весов перед следующим шагом алгоритма;
 - updateWeightsAndBiases(self, learnRate, hGrads, oGrads) - обновление весов сетки;
 - crossEntropyError(self, x_values, t_values) - расчет величины кросс-энтропии;
 - accuracy(self, x_values, t_values) - расчет ошибки в натренированной нейронной сети.
## Результаты экспериментов

| Число нейронов скрытого слоя | К-во эпох | Точность тренировочная | Точность тестовая |
| :---: | :---:  |   :---:     |   :---:   |
|  100  | 14     |    0.9974   |  0.9725   |
|  200  | 17     |    0.9982   |  0.9837   |
|  300  | 22     |    0.9994   |  0.9821   |