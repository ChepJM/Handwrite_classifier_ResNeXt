# cv_classifier

Пайплайн CV классификации изображений с помощью Pytorch

Решение:
1. В EDA.ipynb смотрим распределение классов, практически равномерное за исключением одного. 
Считаем статистики для нормализации по датасету  
2. Тренировка в train.py.  
Так как распределение классов равномерное можно использовать accuracy. Сохраняется лучшая эпоха.  
Датасет должен располагаться в папке input.  
Сохранение модели в model_saved 
3. Высчитывание валидации в test.py  
Полученный результат - accuracy=0.9666