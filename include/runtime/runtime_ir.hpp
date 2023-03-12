#ifndef WBEE_INFRE_RUNTIME_IR_HPP
#define WBEE_INFRE_RUNTIME_IR_HPP

#include <string>
#include <vector>
#include "ir.h"
#include "runtime_operator.hpp"

namespace wbee_infer {

    // 计算图结构,由多个计算节点和节点之间的数据流图组成
    class RuntimeGraph {
    public:
        /**
         * 计算图的初始化
         * @return 是否初始化车工
        */
        bool Init();


        /**
         * constructor
         * @param param_path 计算图的结构文件
         * @param bin_path 计算图中的权重文件
        */
        RuntimeGraph(std::string param_path, std::string bin_path);

        /**
         *  @param bin_path 计算图中的权重文件路径
        */
        void set_bin_path(const std::string &bin_path);

        /**
         *  @param pram_path 计算图中的结构文件路径
        */
        void set_param_path(const std::string &pram_path);

        /**
         * @return 返回结构文件路径
        */
        const std::string &param_path() const;

        /**
         * @return 返回权重文件路径
        */
        const std::string &bin_path() const;

        /**
         * @return 返回std::vector<RuntimeOperator>
        */
        const std::vector<std::shared_ptr<RuntimeOperator>> operators() const;
    
    private:

        /**
         * 初始化wbee infer计算图节点中的输入操作数
         * @param inputs pnnx中的输入操作数
         * @param runtime_operator 计算图节点
         */
        static void InitInputOperators(const std::vector<pnnx::Operand *> &inputs,
                                        const std::shared_ptr<RuntimeOperator> &runtime_operator);


        /**
         * 初始化wbee infer计算图节点中的输出操作数
         * @param outputs pnnx中的输出操作数
         * @param runtime_operator 计算图节点
         */
        static void InitOutputOperators(const std::vector<pnnx::Operand *> &outputs,
                                        const std::shared_ptr<RuntimeOperator> &runtime_operator);



        /**
         * 初始化wbee infer计算图中的节点属性
         * @param attrs pnnx中的节点属性
         * @param runtime_operator 计算图节点
         */
        static void InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                                    const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * 初始化wbee infer计算图中的节点参数
         * @param params pnnx中的参数属性
         * @param runtime_operator 计算图节点
         */
        static void InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                    const std::shared_ptr<RuntimeOperator> &runtime_operator);

    private:
        enum class GraphState {
            NeedInit = -2,
            NeedBuild = -1,
            Complete = 0,
        };

        GraphState graph_state_ = GraphState::NeedInit;
        std::string input_name_; // 计算图输入节点的名称
        std::string output_name_; // 计算图输出节点的名称
        std::string param_path_; // 计算图中的结构文件
        std::string bin_path_; // 计算图的权重文件

        std::map<std::string,std::shared_ptr<RuntimeOperator>> input_operators_maps_; // 保存输入节点
        std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators_maps_; // 保存输出节点
        std::vector<std::shared_ptr<RuntimeOperator>> operators_; /// 计算图的计算节点
        std::unique_ptr<pnnx::Graph> graph_; /// pnnx的graph
    };
}//end namespace
#endif