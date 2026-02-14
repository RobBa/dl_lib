/**
 * @file tensor.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-12-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "dim_type.h"
#include "topological_sort.h"
#include "graph_node.h"

#include "global_params.h"
#include "initializers.h"

#include <memory>
#include <optional>

#include <iostream>

#include <concepts>
#include <cassert>

// break circular dependency
namespace graph {
    class GraphNode;
    class TopologicalSort;
}

enum class Device {
    CPU,
    CUDA
};

constexpr const char* DeviceToString(Device d) {
    switch(d){
        case Device::CPU:
            return "CPU";
        case Device::CUDA:
            return "CUDA";
    }
}

class Tensor final {
    friend class graph::TopologicalSort;

    private:
        /**
         * @brief Here we encapsulate the tensor's values. 
         * Enables us to use a shared_ptr, as well as encapsulate all the 
         * memory management logic for different devices, like a GPU.
         * Structured as a flat array, the logic for multiple dimensions 
         * encapsulated by surrounding tensor object. 
         */
        class tensorValues_t final {
        private:
            tensorSize_t size = 0;
            ftype* values = nullptr;
            
            Device device;
            inline static Device defaultDevice = Device::CPU;

            void addOtherCpu(const tensorValues_t& other) noexcept;

        public:
            explicit tensorValues_t();
            explicit tensorValues_t(Device d);
            ~tensorValues_t() noexcept;

            tensorValues_t(const tensorValues_t& other) = delete;
            tensorValues_t& operator=(const tensorValues_t& other) = delete;

            tensorValues_t(tensorValues_t&& other) noexcept;
            tensorValues_t& operator=(tensorValues_t&& other) noexcept;

            explicit operator bool() const noexcept;
            ftype& operator[](int idx);
            ftype get(int idx) const;

            tensorSize_t getSize() const noexcept;

            // needed for gradient descent
            tensorValues_t& operator+=(const tensorValues_t& other);

            template<typename T>
            requires (std::is_integral_v< std::remove_const_t<T> >)
            void resize(const T size) {
                this->size = static_cast<tensorSize_t>(size);

                switch(this->device){
                    case Device::CPU:
                        values = static_cast<ftype*>( std::malloc(size * sizeof(ftype)) );
                        break;
                    case Device::CUDA:
                        std::__throw_invalid_argument("Not implemented yet.");
                        break;
                }
            }
            
            void setDevice(const Device d) noexcept;
            Device getDevice() const noexcept;

            static void copyValues(tensorValues_t& target, const tensorValues_t& origin);

            static void setDefaultDevice(const Device d) noexcept;
            static Device getDefaultDevice() noexcept;
        };

        Dimension dims;

        std::unique_ptr<tensorValues_t> values = nullptr; // values of tensor TODO: make unique?
        
        bool requiresGrad = false;
        std::shared_ptr<Tensor> grads = nullptr; // gradients
        std::shared_ptr<graph::GraphNode> cgNode = nullptr;
    
        Tensor multiplyScalar(const Tensor& scalar, const Tensor& other) const noexcept;
        void matMul2DCpu(Tensor& res, const Tensor& left, const Tensor& right, const tensorSize_t resOffset, 
                           const tensorSize_t leftOffset, const tensorSize_t rightOffset) const;

        Tensor matMulImpl(const Tensor& left, const Tensor& right) const;

        friend void printValuesCpu(std::ostream& os, const Tensor& t);

        tensorSize_t computeIdx(const std::vector<tensorDim_t>&& idx) const;
        tensorSize_t computeIdx(const std::vector<tensorDim_t>& idx) const;
        tensorSize_t getTotalDimSize(const tensorDim_t dim) const;

        template<typename T> 
        requires (std::is_same_v<std::remove_cv_t<T>, Dimension>)
        explicit Tensor(T&& dims, Device d, bool requiresGrad)
            : Tensor{std::forward<Dimension>(dims.toVector()), tensorValues_t::getDefaultDevice(), requiresGrad}
        { }

    public:

        explicit Tensor(const Dimension& dims, bool requiresGrad=false) :
            dims{dims}, values{std::make_unique<tensorValues_t>()}, requiresGrad{requiresGrad} {            
            values->resize(this->dims.getSize());
        }

        explicit Tensor(const Dimension& dims, Device d, bool requiresGrad=false) :
            dims{dims}, values{std::make_unique<tensorValues_t>(d)}, requiresGrad{requiresGrad} {            
            values->resize(this->dims.getSize());
        }

        /** 
         * Tensors can become very large. Deleting those two
         * helps us to not accidentally copy something we do not 
         * intend to copy. We create extra methods for that.
         */
        Tensor(const Tensor& other) = delete;
        Tensor& operator=(const Tensor& other) = delete;

        Tensor createEmptyCopy() const;
        Tensor createDeepCopy() const;

        /**
         * @brief Moving. Move array of values
         * to new instance as well.
         */
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;

        void reset(const ftype x);
        void reset(const utility::InitClass ic);
        
        const Dimension& getDims() const noexcept;
        tensorSize_t getSize() const noexcept;

        //Tensor operator@(const Tensor& other) const; in higher C++ versions than 20
        Tensor matmul(const Tensor& other) const;

        Tensor operator+(const Tensor& other) const;
        Tensor add(const Tensor& other) const;

        // TODO: Tensor operator-(const Tensor& other) const;

        Tensor operator*(Tensor const& t) const;
        Tensor elementwiseMul(const Tensor& other) const;

        // TODO: Tensor operator/(const Tensor& other) const;

        void backward();

        void transpose() noexcept;
        void transpose(const tensorDim_t dim1, const tensorDim_t dim2) noexcept;
        
        void permute(const std::vector<tensorDim_t>&& newOrder) noexcept;

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t) noexcept;

        // for convenience we provide some simple getters
        ftype get(tensorDim_t idx) const;
        ftype get(tensorDim_t idx0, tensorDim_t idx1) const;
        ftype get(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2) const;
        ftype get(tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3) const;

        ftype get(const std::vector<tensorDim_t>&& idx) const;

        // for convenience we provide some simple setters
        ftype set(ftype item, tensorDim_t idx);
        ftype set(ftype item, tensorDim_t idx0, tensorDim_t idx1);
        ftype set(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2);
        ftype set(ftype item, tensorDim_t idx0, tensorDim_t idx1, tensorDim_t idx2, tensorDim_t idx3);
        void set(ftype item, const std::vector<tensorDim_t>&& idx);

        ftype& operator[](const tensorSize_t idx);

        void setDevice(const Device d) noexcept;
        Device getDevice() const noexcept;

        bool getRequiresGrad() const noexcept { return requiresGrad; }
        void setRequiresGrad(const bool requiresGrad) noexcept { 
            this->requiresGrad=requiresGrad; 
        }

        // these two should not be exposed to the python interface
        static void setDefaultDevice(const Device d) noexcept;
        static Device getDefaultDevice() noexcept;
};