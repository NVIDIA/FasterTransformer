#include <iostream>
#include <vector>
#include <unordered_map>

#include "src/fastertransformer/utils/Tensor.h"

using namespace fastertransformer;

class TestFailureError : public std::exception {
private:
    std::string msg_;
public:
    explicit TestFailureError() = default;
    explicit TestFailureError(std::string name, std::string msg = "") {
        msg_ = fmtstr("TEST FAIL [%s] %s", name.c_str(), msg.c_str());
    }
	const char* what () const throw () {
    	return msg_.c_str();
    }
};

#define EXPECT_TRUE(cond)                           \
    do { if(!(cond)) {                              \
        FT_LOG_ERROR("TEST FAIL [%s] at %s:%d",     \
                     __func__, __FILE__, __LINE__); \
        throw TestFailureError(__func__);           \
    } } while(false)

#define EXPECT_FALSE(cond)                          \
    do { if(cond) {                                 \
        FT_LOG_ERROR("TEST FAIL [%s] at %s:%d",     \
                     __func__, __FILE__, __LINE__); \
        throw TestFailureError(__func__);           \
    } } while(false)

#define EXPECT_EQUAL_TENSORS(t1, t2)       \
    do {                                   \
        EXPECT_TRUE(t1.where == t2.where); \
        EXPECT_TRUE(t1.type == t2.type);   \
        EXPECT_TRUE(t1.shape == t2.shape); \
        EXPECT_TRUE(t1.data == t2.data);   \
    } while(false)

void testTensorMapHasKey() {
    bool* v1 = new bool(true);
    float* v2 = new float[6]{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    Tensor t1 = Tensor{MEMORY_CPU, TYPE_BOOL, {1}, v1};
    Tensor t2 = Tensor{MEMORY_CPU, TYPE_FP32, {3, 2}, v2};

    TensorMap map({{"t1", t1}, {"t2", t2}});
    EXPECT_TRUE(map.isExist("t1"));
    EXPECT_TRUE(map.isExist("t2"));
    EXPECT_FALSE(map.isExist("t3"));

    delete v1;
    delete[] v2;
}

void testTensorMapInsert() {
    int* v1 = new int[4]{1, 10, 20, 30};
    float* v2 = new float[2]{1.0f, 2.0f};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);
    Tensor t2 = Tensor(MEMORY_CPU, TYPE_INT32, {2}, v2);

    TensorMap map({{"t1", t1}});
    EXPECT_TRUE(map.size() == 1);
    EXPECT_TRUE(map.isExist("t1"));
    EXPECT_EQUAL_TENSORS(map.at("t1"), t1);
    EXPECT_FALSE(map.isExist("t2"));

    // forbid a none tensor.
    try {
        map.insert("none", {});
        map.insert("empty", Tensor(MEMORY_CPU, TYPE_INT32, {}, nullptr));
        EXPECT_TRUE(false);
    } catch (std::runtime_error& e) {
        EXPECT_TRUE(true);
    }
    EXPECT_TRUE(map.size() == 1);

    // forbid a duplicated key.
    try {
        map.insert("t1", t2);
        EXPECT_TRUE(false);
    } catch (std::runtime_error& e) {
        EXPECT_TRUE(true);
    }
    EXPECT_TRUE(map.size() == 1);

    map.insert("t2", t2);
    EXPECT_TRUE(map.size() == 2);
    EXPECT_EQUAL_TENSORS(map.at("t2"), t2);

    delete[] v1;
    delete[] v2;
}

void testTensorMapGetVal() {
    int* v1 = new int[4]{1, 10, 20, 30};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);

    TensorMap map({{"t1", t1}});
    EXPECT_TRUE(map.size() == 1);

    try {
        int val = map.getVal<int>("t3");
        EXPECT_TRUE(false);
    } catch(std::runtime_error& e) {
        EXPECT_TRUE(true);
    }
    EXPECT_TRUE(map.getVal<int>("t1") == 1);
    EXPECT_TRUE(map.getVal<int>("t1", 3) == 1);
    EXPECT_TRUE(map.getVal<int>("t2", 3) == 3);

    v1[0] += 1;  // update value.
    EXPECT_TRUE(map.getVal<int>("t1") == 2);
    EXPECT_TRUE(map.getVal<int>("t1", 3) == 2);

    size_t index = 2;
    EXPECT_TRUE(map.getValWithOffset<int>("t1", index) == 20);
    EXPECT_TRUE(map.getValWithOffset<int>("t1", index, 3) == 20);
    EXPECT_TRUE(map.getValWithOffset<int>("t2", index, 3) == 3);
    delete[] v1;
}

void testTensorMapGetTensor() {
    bool* t1_val = new bool(true);
    float* t2_val = new float[6]{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    Tensor t1 = Tensor{MEMORY_CPU, TYPE_BOOL, {1}, t1_val};
    Tensor t2 = Tensor{MEMORY_CPU, TYPE_FP32, {3, 2}, t2_val};

    int* default_val = new int[4]{0, 1, 2, 3};
    Tensor default_tensor = Tensor{MEMORY_CPU, TYPE_INT32, {4}, default_val};

    TensorMap map({{"t1", t1}, {"t2", t2}});

    try {
        Tensor t = map.at("t3");
        EXPECT_TRUE(false);
    } catch(std::runtime_error& e) {
        EXPECT_TRUE(true);
    }
    EXPECT_EQUAL_TENSORS(map.at("t1", default_tensor), t1);
    EXPECT_EQUAL_TENSORS(map.at("t2", default_tensor), t2);
    EXPECT_EQUAL_TENSORS(map.at("t3", default_tensor), default_tensor);
    EXPECT_EQUAL_TENSORS(map.at("t3", {}), Tensor());

    delete[] default_val;
    delete[] t2_val;
    delete[] t1_val;
}

void testEmptyTensorMinMaxRaiseError() {
    Tensor t1;
    try {
        int minval = t1.min<int>();
        int maxval = t1.max<int>();
        EXPECT_TRUE(false);
    } catch (std::runtime_error& e) {
        EXPECT_TRUE(true);
    }

    Tensor t2 = Tensor{MEMORY_CPU, TYPE_INT32, {1}, nullptr};
    try {
        int minval = t2.min<int>();
        int maxval = t2.max<int>();
        EXPECT_TRUE(false);
    } catch (std::runtime_error& e) {
        EXPECT_TRUE(true);
    }
}

template<typename T>
void testTensorMinMax() {
    constexpr int SIZE = 4;
    constexpr T MAX_VAL = T(4);
    constexpr T MIN_VAL = T(1);

    T* v1 = new T[SIZE]{T(1), T(2), T(3), T(4)};
    T* v2 = new T[SIZE]{T(4), T(3), T(2), T(1)};
    T* v3 = new T[SIZE]{T(1), T(2), T(4), T(3)};
    Tensor t1 = Tensor{MEMORY_CPU, getTensorType<T>(), {SIZE}, v1};
    Tensor t2 = Tensor{MEMORY_CPU, getTensorType<T>(), {SIZE}, v2};
    Tensor t3 = Tensor{MEMORY_CPU, getTensorType<T>(), {SIZE}, v3};

    EXPECT_TRUE(t1.max<T>() == MAX_VAL);
    EXPECT_TRUE(t2.max<T>() == MAX_VAL);
    EXPECT_TRUE(t3.max<T>() == MAX_VAL);
    EXPECT_TRUE(t1.min<T>() == MIN_VAL);
    EXPECT_TRUE(t2.min<T>() == MIN_VAL);
    EXPECT_TRUE(t3.min<T>() == MIN_VAL);

    delete[] v1;
    delete[] v2;
    delete[] v3;
}

template<typename T>
void testTensorAny() {
    constexpr int SIZE = 4;
    T* v = new T[SIZE]{T(1), T(2), T(3), T(4)};
    Tensor t = Tensor{MEMORY_CPU, getTensorType<T>(), {SIZE}, v};
    EXPECT_TRUE(t.any<T>(T(1)));
    EXPECT_FALSE(t.any<T>(T(5)));
    delete[] v;
}

template<typename T>
void testTensorAll() {
    constexpr int SIZE = 4;
    T* v1 = new T[SIZE]{T(1), T(1), T(1), T(1)};
    T* v2 = new T[SIZE]{T(1), T(1), T(1), T(2)};
    Tensor t1 = Tensor{MEMORY_CPU, getTensorType<T>(), {SIZE}, v1};
    Tensor t2 = Tensor{MEMORY_CPU, getTensorType<T>(), {SIZE}, v2};
    EXPECT_TRUE(t1.all<T>(T(1)));
    EXPECT_FALSE(t2.all<T>(T(2)));
    delete[] v1;
    delete[] v2;
}


template<typename T>
void testTensorSlice() {
    constexpr int SIZE = 12;
    T* v = new T[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        v[i] = i;
    }
    DataType dtype = getTensorType<T>();
    Tensor t1 = Tensor(MEMORY_CPU, dtype, {3, 4}, v);
    Tensor t2 = t1.slice({2, 4}, 4);
    EXPECT_EQUAL_TENSORS(t2, Tensor(MEMORY_CPU, dtype, {2, 4}, &v[4]));
    try {
        Tensor overflowed_tensor = t1.slice({2, 4}, 5);
        EXPECT_TRUE(false);
    } catch (std::runtime_error& e) {
        EXPECT_TRUE(true);
    }
    delete[] v;
}

int main() {
    testTensorMapHasKey();
    testTensorMapInsert();
    testTensorMapGetVal();
    testTensorMapGetTensor();
    testEmptyTensorMinMaxRaiseError();
    testTensorMinMax<int8_t>();
    testTensorMinMax<int>();
    testTensorMinMax<float>();
    testTensorAny<int8_t>();
    testTensorAny<int>();
    testTensorAny<float>();
    testTensorAll<int8_t>();
    testTensorAll<int>();
    testTensorAll<float>();
    testTensorSlice<int8_t>();
    testTensorSlice<int>();
    testTensorSlice<float>();
    FT_LOG_INFO("Test Done");
    return 0;
}
