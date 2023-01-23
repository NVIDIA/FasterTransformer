#ifndef CONV1x1_INTERFACE_HPP_
#define CONV1x1_INTERFACE_HPP_

class Conv1x1Interface 
{
public:
    virtual void run(uint8_t* D, uint8_t* A, uint8_t* B, uint16_t* bias, float ab_scale, float d_scale, int N, int H, int W, int C, int K) = 0;
};

#endif // CONV1x1_INTERFACE_HPP_