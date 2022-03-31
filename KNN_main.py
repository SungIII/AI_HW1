#B711093 성의현
import KNN_class
import numpy as np
from sklearn.datasets import load_iris

#데이터 불러오기
iris = load_iris()

#학습용 데이터와 학습이 잘 되었는지 확인하는 확인용 데이터를 분리
learn_data = np.empty((0, 4)) #학습용 데이터의 4가지 속성값 저장
learn_data_target = np.array([], dtype='int32') #learn_data와 같은 index의 값이 해당 iris의 종류(string으로 저장)
test_data = np.empty((0, 4)) #확인용 데이터의 4가지 속성값 저장
test_data_target = np.array([], dtype='int32') #test_data와 같은 index의 값이 해당 iris의 종류(string으로 저장)
target_names = iris.target_names #target의 이름을 담은 array저장

for i in range(0,150) : #150개의 데이터 추출
    if(i % 10) :
        learn_data = np.append(learn_data, np.array([iris.data[i]]), axis=0)
        learn_data_target = np.append(learn_data_target, iris.target[i])
    else : #10개 중에 하나꼴로 test데이터에 추출
        test_data = np.append(test_data, np.array([iris.data[i]]), axis=0)
        test_data_target = np.append(test_data_target, iris.target[i])

print(type(learn_data_target[3]))

# k = 3인 경우
iris_3 = KNN_class.KNN(learn_data, learn_data_target, target_names, 3)
print("data test for k = 3 in Majority vote way>>")
for i in range(0, len(test_data)) :
    print("Test Data Index : " + str(i) + " // Computed class : " + iris_3.test(test_data[i], 'vote') + " // True class : " + target_names[test_data_target[i]])
print("\ndata test for k = 3 in Weighted majority vote way>>")
for i in range(0, len(test_data)) :
    print("Test Data Index : " + str(i) + " // Computed class : " + iris_3.test(test_data[i], 'w_vote') + " // True class : " + target_names[test_data_target[i]])
print("-------------------------------------------------------------------------------\n")

# k = 5인 경우
iris_5 = KNN_class.KNN(learn_data, learn_data_target, target_names, 5)
print("data test for k = 5 in Majority vote way>>")
for i in range(0, len(test_data)) :
    print("Test Data Index : " + str(i) + " // Computed class : " + iris_5.test(test_data[i], 'vote') + " // True class : " + target_names[test_data_target[i]])
print("\ndata test for k = 5 in Weighted majority vote way>>")
for i in range(0, len(test_data)) :
    print("Test Data Index : " + str(i) + " // Computed class : " + iris_5.test(test_data[i], 'w_vote') + " // True class : " + target_names[test_data_target[i]])
print("-------------------------------------------------------------------------------\n")

# k = 9인 경우
iris_9 = KNN_class.KNN(learn_data, learn_data_target, target_names, 9)
print("data test for k = 9 in Majority vote way>>")
for i in range(0, len(test_data)) :
    print("Test Data Index : " + str(i) + " // Computed class : " + iris_9.test(test_data[i], 'vote') + " // True class : " + target_names[test_data_target[i]])
print("\ndata test for k = 9 in Weighted majority vote way>>")
for i in range(0, len(test_data)) :
    print("Test Data Index : " + str(i) + " // Computed class : " + iris_9.test(test_data[i], 'w_vote') + " // True class : " + target_names[test_data_target[i]])
print("\n")