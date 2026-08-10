/**
 * @file threadpool.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-08-10
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "utility/utils.h"

#include <vector>
#include <queue>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <functional>
#include <algorithm>

namespace threadpool_impl {
    class ThreadPool final {
        private:
            std::vector<std::thread> threads;
            std::queue<std::function<void()>> tasks;

            std::mutex m;
            std::condition_variable cond;
            std::atomic<bool> stop;
            std::atomic<int> activeTasks;

            void workerLoop() {
                while(true) {
                    std::function<void()> task;

                    // block until task received, then start execution
                    {
                        std::unique_lock<std::mutex> lock(m);
                        cond.wait(lock, 
                            [this] { 
                                return stop || !tasks.empty(); // while(![predicate of this line]) { wait(lock); }
                            }
                        );

                        if(stop && tasks.empty()) { // for clean destructor
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                        activeTasks++;
                    }

                    task();
                    
                    {
                        // needs lock to coordinate with waitAll()
                        std::unique_lock<std::mutex> lock(m);
                        activeTasks--;
                    }
                    cond.notify_all(); // notifies waitAll()
                }
            }


        public:
            ThreadPool() : stop{false}, activeTasks{0} {
                const unsigned int nthreads = std::max(static_cast<unsigned int>(1), std::thread::hardware_concurrency() / 2);
                threads.reserve(nthreads);
                for(unsigned int i = 0; i < nthreads; i++) {
                    threads.emplace_back(&ThreadPool::workerLoop, this);
                }
            };

            ~ThreadPool() noexcept {
                stop = true;
                cond.notify_all();

                for(auto& t: threads) {
                    if(t.joinable()) {
                        t.join();
                    }
                }
            }

            ThreadPool(const ThreadPool&) = delete;
            ThreadPool& operator=(const ThreadPool&) = delete;

            ThreadPool(ThreadPool&&) = delete;
            ThreadPool& operator=(ThreadPool&&) = delete;

            void enque(std::function<void()> f) {
                std::unique_lock<std::mutex> lock(m);            
                tasks.push(std::move(f));
                cond.notify_one();
            } 

            void waitAll() {
                std::unique_lock<std::mutex> lock(m);
                cond.wait(lock, 
                    [this] {
                        return tasks.empty() && activeTasks == 0;
                    }
                );
            }
    };
}

namespace threadpool {
  inline static threadpool_impl::ThreadPool threadPool;
}