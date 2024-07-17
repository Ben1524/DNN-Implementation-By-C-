 
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
 
namespace NN
{
    class Neuron;
 
    class Network;
 
    typedef std::map<Neuron *, double> Links;
 
 
    class ActivationFunction // 激活函数
    {
    public:
 
        virtual double Output(double x) = 0;
 
        virtual double der(double x) = 0; // 导数
    };
 
    class LossFunction // 计算损失函数
    {
    public:
 
        virtual double Error(double output, double target) = 0;
 
        virtual double der(double output, double target) = 0;
 
    protected:
 
    };
 
    class ReLUFunction : public ActivationFunction // ReLU激活函数 f(x)=max(0,x)
    {
    public:
        virtual double Output(double x)
        {
            return x > 0 ? x : 0;
        }
 
        virtual double der(double x)
        {
            return x > 0 ? 1 : 0;
        }
    };
 
    class MSEFunction : public LossFunction // 均方误差损失函数
    {
    public:
        virtual double Error(double output, double target)
        {
            return 0.5 * (output  target) * (output  target);
        }
 
        virtual double der(double output, double target)
        {
            return output  target;
        }
    };
 
 
    class Neuron // 神经元
    {
    public:
        Neuron(ActivationFunction *funcActivation = nullptr, LossFunction *funcComputeLoss = nullptr);
        virtual ~Neuron();
        virtual bool Forward(); // 前向传播
        virtual double GetOutput(); // 获取输出
        virtual double GetNetOutput(); // 获取加权输入
        virtual bool LinkNeuron(Neuron *pInput); // 连接神经元
        virtual bool LinkNeuron(std::vector<Neuron *> &pInputs);
        virtual bool LinkNeuron(Neuron *pInput, double fweight);
        virtual bool SetNetOutput(double fOutPut); // 设置加权输入
        virtual bool UpdateParamters(double fLearnRateing); // 更新参数
        virtual double GetWeight(Neuron *pInput); // 获取权重
        virtual void SetWeight(Neuron *pInput, double fWeight); // 设置权重
        virtual void SetBiases(double fBiases); // 设置偏置
    public:
        double m_fBiases; // 偏置
        double m_fOutput; // 输出
        double m_fNetOutput; // 加权输入
        double m_fdelta; // 误差
        Links m_Links; // 连接,一个神经元可以连接多个神经元
        ActivationFunction *m_funcActivation; // 激活函数
        LossFunction *m_funcComputeLoss;    // 损失函数
    };
    typedef std::vector<Neuron *> NetworkLayer; // 神经各种网络层
 
    class Network {
    public:
        Network(int InputSize, double fLearnRating = 0.0000001); // 构造函数.输入层大小,学习率
        Network(int InputSize, double fLearnRating, ActivationFunction *funcActivation, LossFunction *funcComputeLoss);
 
        virtual ~Network();
        bool AddLayer(int NeuronSize); // 添加层,神经元个数
        bool AddLayerNode(int nIndex, int NeuronSize = 1);
        size_t GetLayerSize();
        size_t GetLayerNodeSize(int nIndex);
        Neuron *GetNeuron(int nLayerIndex, int nNeuronIndex);
        virtual bool Forward(std::vector<double> InputData, std::vector<double> &OutputData); // 前向传播,输入数据,输出数据
        virtual bool Back(std::vector<double> &Target); // 反向传播,目标数据
        virtual double GetLossError(std::vector<double> &Target);
    protected:
        std::vector<NetworkLayer> m_vecLayer; // 神经网络层
        double m_fLearnRating;  // 学习率
        ActivationFunction *m_funcActivation;
        LossFunction *m_funcComputeLoss;
    };
}
static NN::ReLUFunction gst_ActivationFunction; // ReLU激活函数
static NN::MSEFunction gst_ComputeLossFunction; // 均方误差损失函数
 
NN::Network::Network(int InputSize, double fLearnRating)
        : m_fLearnRating(fLearnRating)
{
    m_funcActivation = &gst_ActivationFunction;
    m_funcComputeLoss = &gst_ComputeLossFunction;
    AddLayer(InputSize);
}
 
NN::Network::Network(int InputSize, double fLearnRating, ActivationFunction *funcActivation, LossFunction *funcComputeLoss)
        : m_fLearnRating(fLearnRating)
{
    m_funcActivation = funcActivation;
    m_funcComputeLoss = funcComputeLoss;
    AddLayer(InputSize);
}
 
NN::Network::~Network()
{
    for (auto &L: m_vecLayer) {
        for (auto &n: L) {
            delete n;
        }
    }
    m_vecLayer.clear();
}
 
bool NN::Network::AddLayer(int NeuronSize)
{
    NetworkLayer NewLayer;
    size_t nLayerSize = GetLayerSize();
    for (int i = 0; i < NeuronSize; ++i) {
        Neuron *pNeuron = new Neuron(m_funcActivation, m_funcComputeLoss); // 创建神经元
        NewLayer.push_back(pNeuron);
        if (nLayerSize != 0) {  // 如果不是输入层
            pNeuron>LinkNeuron(m_vecLayer[nLayerSize  1]); // 连接到前一层
        }
    }
    m_vecLayer.push_back(NewLayer);
    return false;
}
 
bool NN::Network::AddLayerNode(int nIndex, int NeuronSize)
{
    if (nIndex < 0 || nIndex >= GetLayerSize())
        return false;
    for (int i = 0; i < NeuronSize; ++i) {
        Neuron *pNeuron = new Neuron(m_funcActivation, m_funcComputeLoss);
        m_vecLayer[nIndex].push_back(pNeuron);
        //连接前面的层
        if (nIndex  1 >= 0) {
            pNeuron>LinkNeuron(m_vecLayer[nIndex  1]);
        }
        //链接到后面的层
        if (nIndex + 1 < GetLayerSize()) {
            for (auto &LN: m_vecLayer[nIndex + 1]) {
                LN>LinkNeuron(pNeuron);
            }
        }
    }
    return true;
}
 
size_t NN::Network::GetLayerSize()
{
    return m_vecLayer.size();
}
 
size_t NN::Network::GetLayerNodeSize(int nIndex)
{
    if (nIndex < 0 || nIndex >= GetLayerSize())
        return 0;
    return m_vecLayer[nIndex].size();
}
 
NN::Neuron *NN::Network::GetNeuron(int nLayerIndex, int nNeuronIndex)
{
    if (nLayerIndex < 0 || nLayerIndex >= GetLayerSize())
        return nullptr;
 
    if (nNeuronIndex < 0 || nNeuronIndex >= GetLayerNodeSize(nLayerIndex))
        return nullptr;
 
    return m_vecLayer[nLayerIndex][nNeuronIndex];
}
 
bool NN::Network::Forward(std::vector<double> InputData, std::vector<double> &OutputData)
{
    if (GetLayerSize() <= 0 || InputData.size() != m_vecLayer[0].size())
        return false;
 
    //更新到输入层
    for (int i = 0; i < InputData.size(); ++i)
        m_vecLayer[0][i]>SetNetOutput(InputData[i]);
 
    //第二层开始推算
    for (int i = 1; i < GetLayerSize(); ++i)
        for (int j = 0; j < m_vecLayer[i].size(); ++j)
            m_vecLayer[i][j]>Forward();
 
    //结果
    if (GetLayerSize() >= 1) {
        NetworkLayer &OutputLayer = m_vecLayer[GetLayerSize()  1];
        for (int i = 0; i < OutputLayer.size(); ++i)
            OutputData.push_back(OutputLayer[i]>GetOutput());
    }
    return true;
}
 
bool NN::Network::Back(std::vector<double> &Target)
{
    if (GetLayerSize() <= 0 || Target.size() != m_vecLayer[GetLayerSize()  1].size())
        return false;
 
    //计算输出层误差（Compute Output Layer Error）
    NetworkLayer &OutputLayer = m_vecLayer[GetLayerSize()  1];
    for (int i = 0; i < OutputLayer.size(); ++i) {
        OutputLayer[i]>m_fdelta = m_funcComputeLoss>der(OutputLayer[i]>m_fOutput,
                                                          Target[i])/**m_funcActivation>der(OutputLayer[i]>m_fNetOutput)*/;
    }
 
    //反向传播误差（Backpropagate Error）
    for (int i = m_vecLayer.size()  2; i > 0; i) {
        NetworkLayer &NowLayer = m_vecLayer[i];
        NetworkLayer &NextLayer = m_vecLayer[i + 1];
 
        for (auto &now: NowLayer) {
            //该神经元连接到输出层神经元的权重和对应的输出层神经元的误差项之和
            double ftemp = 0.0;
            for (auto &next: NextLayer) {
                ftemp += next>GetWeight(now) * next>m_fdelta;
            }
 
            now>m_fdelta = m_funcActivation>der(now>m_fNetOutput) * ftemp;
        }
    }
 
    //计算梯度和更新参数
    for (int i = 1; i < m_vecLayer.size(); ++i) {
        NetworkLayer &NowLayer = m_vecLayer[i];
        for (auto &node: NowLayer) {
            node>UpdateParamters(m_fLearnRating);
        }
    }
    return false;
}
 
double NN::Network::GetLossError(std::vector<double> &Target)
{
    if (GetLayerSize() <= 0 || Target.size() != m_vecLayer[GetLayerSize()  1].size())
        return 0.0;
 
    double fLossError = 0.0;
    NetworkLayer &OutputLayer = m_vecLayer[GetLayerSize()  1];
    for (int i = 0; i < OutputLayer.size(); ++i) {
        fLossError += m_funcComputeLoss>Error(OutputLayer[i]>m_fOutput, Target[i]);
    }
    return fLossError;
}
 
NN::Neuron::Neuron(ActivationFunction *funcActivation, LossFunction *funcComputeLoss)
        : m_fBiases(0.1), m_fOutput(0.0), m_fNetOutput(0.0), m_fdelta(0.0), m_funcActivation(funcActivation), m_funcComputeLoss(funcComputeLoss)
{
    if (!m_funcActivation)
        m_funcActivation = &gst_ActivationFunction;
    if (!m_funcComputeLoss)
        m_funcComputeLoss = &gst_ComputeLossFunction;
}
 
NN::Neuron::~Neuron()
{
}
 
bool NN::Neuron::Forward()
{
    //加权输入
    m_fNetOutput = m_fBiases;
    for (auto it: m_Links) {
        m_fNetOutput += it.first>GetOutput() * it.second;
    }
    m_fOutput = gst_ActivationFunction.Output(m_fNetOutput);
    return true;
}
 
double NN::Neuron::GetOutput()
{
    return m_fOutput;
}
 
double NN::Neuron::GetNetOutput()
{
    return m_fNetOutput;
}
 
bool NN::Neuron::LinkNeuron(Neuron *pInput)
{
    const int weight_base = 100000000;
    double fweight = rand() % weight_base;
    fweight /= weight_base;
    //fweight = 1;
    return LinkNeuron(pInput, fweight);
}
 
bool NN::Neuron::LinkNeuron(std::vector<Neuron *> &pInputs)
{
    for (auto &pInput: pInputs) {
        LinkNeuron(pInput);
    }
    return true;
}
 
bool NN::Neuron::LinkNeuron(Neuron *pInput, double fweight)
{
    //已存在 覆盖
    m_Links[pInput] = fweight;
    return true;
}
 
bool NN::Neuron::SetNetOutput(double fOutPut)
{
    m_fNetOutput = fOutPut;
    m_fOutput = gst_ActivationFunction.Output(fOutPut);
    return true;
}
 
bool NN::Neuron::UpdateParamters(double fLearnRateing)
{
    //更新偏置
    std::cout << "update bias:" << m_fdelta << std::endl;
    m_fBiases = fLearnRateing * m_fdelta;
 
    //更新输入权重
    for (auto &it: m_Links) {
        std::cout<<"update weight:"<<it.second<<std::endl;
        it.second = fLearnRateing * (it.first>GetOutput() * m_fdelta);
    }
    return true;
}
 
double NN::Neuron::GetWeight(Neuron *pInput)
{
    Links::iterator it = m_Links.find(pInput);
    if (m_Links.end() == it)
        return 0.0;
    return it>second;
}
 
void NN::Neuron::SetWeight(Neuron *pInput, double fWeight)
{
    Links::iterator it = m_Links.find(pInput);
    if (m_Links.end() == it)
        return;
    it>second = fWeight;
}
 
void NN::Neuron::SetBiases(double fBiases)
{
    m_fBiases = fBiases;
}
 
 
 
 
int main()
{
    double flearing = 0.0001;
    NN::Network myNet(1, flearing);
    myNet.AddLayer(1);
 
    int max_loss_cnt = 0;
    int e = 0;
    auto calFunc=[](double x){return 0.5 * x + 5;};
    while (true) {
        ++e;
        double x1 = rand() % 10;
        std::vector<double> vecInput({x1});
        std::vector<double> vecTarget({calFunc(x1)});
        std::vector<double> vecOutput;
        myNet.Forward(vecInput, vecOutput);
        double floss = myNet.GetLossError(vecTarget);
        std::cout << "e:" << e << ",lose:" << floss << ",Output:" << vecOutput[0] << ",Target:" << vecTarget[0] << std::endl;
 
        if (floss > 0.00000001) {
            max_loss_cnt = 0;
        } else {
            max_loss_cnt++;
            if (max_loss_cnt > 10)
                break;
        }
        myNet.Back(vecTarget);
    }
 
    for (int i = 0; i < 3; ++i) {
        double x1 = rand() % 10;
        std::vector<double> vecInput({x1});
        std::vector<double> vecTarget({calFunc(x1)});
        std::vector<double> vecOutput;
        myNet.Forward(vecInput, vecOutput);
        double floss = myNet.GetLossError(vecTarget);
        std::cout << "e:" << e << ",lose:" << floss << ",Output:" << vecOutput[0] << ",Target:" << vecTarget[0] << std::endl;
 
    }
    return 0;
}
