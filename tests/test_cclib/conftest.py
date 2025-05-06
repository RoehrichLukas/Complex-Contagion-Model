import pytest

@pytest.fixture
def Non_Int():
    return [0.1,'test',[0,1,2],(0,1),{0,1},{0:1}]

@pytest.fixture
def Non_Pos():
    return [-2,-0.0000000001]

@pytest.fixture
def Non_Prob():
    return [-0.000000000001,1.0000000000001,-5,5]

@pytest.fixture
def Non_Bool():
    return [0.5,-2,'test',[True,False],(True,False),{0,1},{0:1}]

@pytest.fixture
def Non_Int_or_Float():
    return ['test',(1,2),[1,2],{1:2}]

@pytest.fixture
def Non_Array():
    return [1,2.3,-3,True,'test',{1,2},{1:2,2:3}]

@pytest.fixture
def Non_Graph():
    return [1,0.4,-3,[1,2,3],True,'test',{0,1,2},{0:1,2:3}]

@pytest.fixture
def Non_Str():
    return [1,0.4,-3,[1,2,3],True,{0,1,2},{0:1,2:3}]

