//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "omp.h"
#include "Yaml.hpp"
#include <Eigen/Core>
#include "BasicEigenTypes.hpp"

namespace raisim {

template<class ChildEnvironment>
class VectorizedEnvironment_for_test {

 public:

  explicit VectorizedEnvironment_for_test(std::string resourceDir, std::string cfg, bool normalizeObservation=true)
      : resourceDir_(resourceDir), cfgString_(cfg), normalizeObservation_(normalizeObservation) {
    Yaml::Parse(cfg_, cfg);
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
  }

  ~VectorizedEnvironment_for_test() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  void init() {
    omp_set_num_threads(cfg_["num_threads"].template As<int>());
    num_envs_ = cfg_["num_envs"].template As<int>();

    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
    }

    setSeed(0);

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(environments_[0]->getObDim() == 0 || environments_[0]->getActionDim() == 0, "Observation/Action dimension must be defined in the constructor of each environment!")

    /// ob scaling
    if (normalizeObservation_) {
      obMean_.setZero(obDim_);
      obVar_.setOnes(obDim_);
      recentMean_.setZero(obDim_);
      recentVar_.setZero(obDim_);
      delta_.setZero(obDim_);
      epsilon.setZero(obDim_);
      epsilon.setConstant(1e-8);
    }
  }

  // resets all environments and returns observation
  void reset() {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));
    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(ob);
  }

  void observe2(Eigen::Ref<EigenRowMajorMat> &obO) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe2(obO.row(i));
    if (normalizeObservation_)
      updateObservationOStatisticsAndNormalize(obO);
  }

  void step2(Eigen::Ref<EigenRowMajorMat> &action,
             Eigen::Ref<EigenRowMajorMat> &actionO) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep2(i, action, actionO);
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }
  void setObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    obMean_ = mean; obVar_ = var; obCount_ = count; }
  void setObOStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    obMeanO_ = mean; obVarO_ = var; obCountO_ = count; }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      terminalState[i] = environments_[i]->isTerminalState();
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return environments_[0]->getObDim(); }
  int getActionDim() { return environments_[0]->getActionDim(); }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

 private:
  void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      ob.row(i) = (ob.row(i) - obMean_.transpose()).template cwiseQuotient<>((obVar_ + epsilon).cwiseSqrt().transpose());
  }

  void updateObservationOStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      ob.row(i) = (ob.row(i) - obMeanO_.transpose()).template cwiseQuotient<>((obVarO_ + epsilon).cwiseSqrt().transpose());
  }

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action) {
    environments_[agentId]->step(action.row(agentId));
    if (environments_[agentId]->isTerminalState()) {
      environments_[agentId]->reset();
    }
  }
  inline void perAgentStep2(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenRowMajorMat> &actionO) {
    environments_[agentId]->step2(action.row(agentId), actionO.row(agentId));
    if (environments_[agentId]->isTerminalState()) {
      environments_[agentId]->reset();
    }
  }

  std::vector<ChildEnvironment *> environments_;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;

  bool normalizeObservation_ = true;
  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  bool recordVideo_=false, render_=false;
  EigenVec obMean_, obMeanO_;
  EigenVec obVar_, obVarO_;
  float obCount_ = 1e-4;
  float obCountO_ = 1e-4;
  EigenVec recentMean_, recentVar_, delta_;
  EigenVec epsilon;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
