//
// Created by Ryan Deng on 5/7/23.
//

#pragma once

#include <array>
#include <atomic>
#include <mpi.h>
#include <map>
#include <cassert>
#include <chrono>
#include <set>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include "toml.hpp"

struct StencilMDConfig {
    // controls the behavior of the simulation
    bool ONLY_RUN_LAMMPS = false;
    bool ONLY_RUN_STENCIL_MD = false;
    int EXPERIMENT_TYPE = -1;
    bool DEBUG_SEND_RECV_DATA = false;
    bool TEST_AGAINST_LAMMPS = true;
    bool TIME_STENCIL_MD = false;
    bool USE_NEWTON = false;

    // parameters of the simulation
    int NUM_TIMESTEPS_IN_PARALLEL = -1;
    double CUTOFF_RADIUS = 0.0;
    double SKIN_DISTANCE = 0.0;
    double STENCIL_MD_SLOPE = 0.0;

    int NUM_CUTS_X = 0;
    int NUM_CUTS_Y = 0;
    int NUM_CUTS_Z = 0;

    int NUM_ZOIDS_X = 0;
    int NUM_ZOIDS_Y = 0;
    int NUM_ZOIDS_Z = 0;
};

class StencilMDConfigManager {
private:
    StencilMDConfig config;
    bool is_loaded = false;
    static inline std::unique_ptr<StencilMDConfigManager> instance;
    static inline std::mutex init_mutex;

    // Private constructor to prevent external instantiation
    StencilMDConfigManager() = default;

public:
    // The static method to get the single instance
    static StencilMDConfigManager& get_instance() {
        // Double-checked locking pattern (optional, simple lock is often fine)
        if (!instance) {
            std::lock_guard<std::mutex> lock(init_mutex);
            if (!instance) {
                // Use 'new' directly because make_unique can't access private constructor
                instance.reset(new StencilMDConfigManager());
            }
        }
        return *instance;
    }

    // Accessor for the configuration data
    const StencilMDConfig& get_config() const {
        if (!is_loaded) {
            throw std::runtime_error("Configuration accessed before loading.");
        }
        return config;
    }

    void load_from_config_file(const std::string& config_filename) {
        auto toml_config = toml::parse_file(config_filename);

        assert(toml_config["ONLY_RUN_LAMMPS"].value<bool>().has_value());
        assert(toml_config["ONLY_RUN_STENCIL_MD"].value<bool>().has_value());
        assert(toml_config["DEBUG_SEND_RECV_DATA"].value<bool>().has_value());
        assert(toml_config["EXPERIMENT_TYPE"].value<int>().has_value());
        assert(toml_config["USE_NEWTON"].value<bool>().has_value());

        config.ONLY_RUN_LAMMPS = toml_config["ONLY_RUN_LAMMPS"].value<bool>().value();
        config.ONLY_RUN_STENCIL_MD = toml_config["ONLY_RUN_STENCIL_MD"].value<bool>().value();
        config.DEBUG_SEND_RECV_DATA = toml_config["DEBUG_SEND_RECV_DATA"].value<bool>().value();
        config.EXPERIMENT_TYPE = toml_config["EXPERIMENT_TYPE"].value<int>().value();
        config.USE_NEWTON = toml_config["USE_NEWTON"].value<bool>().value();

        assert(toml_config["NUM_TIMESTEPS_IN_PARALLEL"].value<int>().has_value());
        assert(toml_config["CUTOFF_RADIUS"].value<double>().has_value());
        assert(toml_config["SKIN_DISTANCE"].value<double>().has_value());

        config.NUM_TIMESTEPS_IN_PARALLEL = toml_config["NUM_TIMESTEPS_IN_PARALLEL"].value<int>().value();
        config.CUTOFF_RADIUS = toml_config["CUTOFF_RADIUS"].value<double>().value();
        config.SKIN_DISTANCE = toml_config["SKIN_DISTANCE"].value<double>().value();
        config.STENCIL_MD_SLOPE = config.CUTOFF_RADIUS + config.SKIN_DISTANCE + ADDITIONAL_CUTOFF;

        assert(toml_config["NUM_CUTS_X"].value<int>().has_value());
        assert(toml_config["NUM_CUTS_Y"].value<int>().has_value());
        assert(toml_config["NUM_CUTS_Z"].value<int>().has_value());
        config.NUM_CUTS_X = toml_config["NUM_CUTS_X"].value<int>().value();
        config.NUM_CUTS_Y = toml_config["NUM_CUTS_Y"].value<int>().value();
        config.NUM_CUTS_Z = toml_config["NUM_CUTS_Z"].value<int>().value();
        config.NUM_ZOIDS_X = config.NUM_CUTS_X * 2;
        config.NUM_ZOIDS_Y = config.NUM_CUTS_Y * 2;
        config.NUM_ZOIDS_Z = config.NUM_CUTS_Z * 2;

        is_loaded = true;
    }
};

