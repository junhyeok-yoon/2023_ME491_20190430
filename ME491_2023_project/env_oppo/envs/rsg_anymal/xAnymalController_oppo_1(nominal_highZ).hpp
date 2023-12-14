// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include <deque>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

/// change the class name and file name ex) AnymalController_00000000 -> AnymalController_STUDENT_ID
class AnymalController_20002000 {

 public:
  inline bool create(raisim::World *world) {
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));
    anymalO_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(opponentName_));
    world_ = reinterpret_cast<raisim::World *>(world);
    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 95;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);

    //
    timer_ = 0;
    xyBefore_.setZero();
    RR_.setZero();
    contactInfo_.setZero(15);
    discountFactorForSatefy_ = 1.0;
    targetPos_.setZero();
    targetSpeed_ = 0.0;

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    return true;
  }

  inline bool init(raisim::World *world) {
    return true;
  }

  double square(double a){
    return a * a;
  }

  Eigen::Vector3d quatToRPY(const Eigen::Vector4d& q) {
    Eigen::Vector3d rpy;
    const double as = std::min(-2. * (q[1] * q[3] - q[0] * q[2]), .99999);
    rpy(2) = std::atan2(2 * (q[1] * q[2] + q[0] * q[3]),
                        square(q[0]) + square(q[1]) - square(q[2]) - square(q[3]));
    rpy(1) = std::asin(as);
    rpy(0) = std::atan2(2 * (q[2] * q[3] + q[0] * q[1]),
                        square(q[0]) - square(q[1]) - square(q[2]) + square(q[3]));
    return rpy; 
  }

  inline bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    timer_++;
    actionHistory_.push_front(action);
    actionHistory_.pop_back();
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal_->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  Eigen::Vector2d getTargetPos(){
    return targetPos_;
  }

  inline bool reset(raisim::World *world, double theta, double minObjSize = 0.15, int objIdx = -1, raisim::SingleBodyObject* objPtr = NULL) {
    timer_ = 0;
    RR_.setZero();
    const double initR = 1.3*(1+uniDist_(gen_));
    const double initTheta = uniDist_(gen_)*2*M_PI;
    gc_init_.head(2) = Eigen::Vector2d(initR*cos(initTheta), initR*sin(initTheta));
    const double yaw = uniDist_(gen_)*2*M_PI;
    gc_init_.segment(3, 4) << cos(yaw / 2), 0, 0, sin(yaw / 2);
    anymal_->setState(gc_init_, gv_init_);
    anymal_->getState(gc_, gv_);
    
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_init_[3];
    quat[1] = gc_init_[4];
    quat[2] = gc_init_[5];
    quat[3] = gc_init_[6];
    raisim::quatToRotMat(quat, rot);
    rot_ = rot.e();

    // reset states
    anymal_->getState(gc_, gv_);

    // set target speed(0~2)
    targetSpeed_ = 1.25 + 1.25*uniDist_(gen_);
    

    // set target
    double targetR = 1.3*(1+uniDist_(gen_));
    double targetTheta = uniDist_(gen_)*2*M_PI;
    targetPos_ = Eigen::Vector2d(targetR*cos(targetTheta), targetR*sin(targetTheta));
    while((targetPos_ - gc_.head(2)).norm() < 1){
      targetR = 1.3*(1+uniDist_(gen_));
      targetTheta = uniDist_(gen_)*2*M_PI;
      targetPos_ = Eigen::Vector2d(targetR*cos(targetTheta), targetR*sin(targetTheta));
    }
    const Eigen::Vector3d relativeTarget = Eigen::Vector3d(targetPos_(0) - gc_(0), targetPos_(1) - gc_(1), 0);
    const double targetYaw = atan2(relativeTarget(1), relativeTarget(0)) - atan2(rot_(1,0), rot_(0,0)); // my yaw - me to opponent yaw
    Eigen::Matrix3d projectedRot;
    projectedRot << cos(targetYaw), -sin(targetYaw), 0,
                    sin(targetYaw), cos(targetYaw), 0,
                    0, 0, 1;
    Eigen::VectorXd state(41);
    bodyLinearVel_  = rot_.transpose() * gv_.head(3);
    bodyAngularVel_ = rot_.transpose() * gv_.segment(3, 3);
    state << // my info(37)
             gc_.head(3), /// body pose
             3.0 - gc_.head(2).norm(), /// distance to the edge
             rot_.row(2).transpose(), /// body orientation
             gc_.tail(12), /// joint angles
             bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
             gv_.tail(12), /// joint velocity
             // target(4)
             targetPos_, /// opponent position
             (projectedRot*relativeTarget).head(2), /// opponent position relative to me

    myHistory_.clear();
    Eigen::VectorXf oppoState(6); 
    for(int i = 0; i < 10; i++){
      myHistory_.push_front(state);
    }
    actionHistory_.clear();
    Eigen::VectorXf action(12); action.setZero();
    for(int i = 0; i < 5; i++){
      actionHistory_.push_front(action);
    }

    return true;
  }

  void updateOnlyState(){
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    rot_ = rot.e();
  }

  inline void updateObservation(raisim::World *world) {
    updateOnlyState();

    const Eigen::Vector3d relativeTarget = Eigen::Vector3d(targetPos_(0) - gc_(0), targetPos_(1) - gc_(1), 0);
    const double yaw = atan2(relativeTarget(1), relativeTarget(0)) - atan2(rot_(1,0), rot_(0,0)); // my yaw - me to opponent yaw
    Eigen::Matrix3d projectedRot;
    projectedRot << cos(yaw), -sin(yaw), 0,
                    sin(yaw), cos(yaw), 0,
                    0, 0, 1;
    Eigen::VectorXd state(41);
    bodyLinearVel_  = rot_.transpose() * gv_.head(3);
    bodyAngularVel_ = rot_.transpose() * gv_.segment(3, 3);
    state << // my info(37)
             gc_.head(3), /// body pose
             3.0 - gc_.head(2).norm(), /// distance to the edge
             rot_.row(2).transpose(), /// body orientation
             gc_.tail(12), /// joint angles
             bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
             gv_.tail(12), /// joint velocity
             // target(4)
             targetPos_, /// opponent position
             (projectedRot*relativeTarget).head(2), /// opponent position relative to me

    obDouble_ << state, myHistory_[0], actionHistory_[0].cast<double>(), float(targetSpeed_);

    // queue, pop
    myHistory_.push_front(state);
    myHistory_.pop_back();
    // opponentHistory_.push_front(gcOppo_.head(3).cast<float>());
    // opponentHistory_.pop_back();
  }

  void desiredReward(Reward *rewards){
    float actionSmooth = (actionHistory_[0] - actionHistory_[1]).squaredNorm()*(timer_ > 1);
    actionSmooth += (actionHistory_[0] - 2 * actionHistory_[1] + actionHistory_[2]).squaredNorm()*(timer_ > 2);
    rewards->record("f_smooth", double(actionSmooth));
  }

  inline void recordReward(Reward *rewards) {
    updateOnlyState();
    rewards->record("b_torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards->record("a_forwardVel", 5*pow(targetSpeed_*std::min(timer_/200.0, 1.0) - bodyLinearVel_.head(2).norm(), 2) + gv_.segment(3,3).squaredNorm());
    RR_ << gc_.head(2).norm(), targetPos_.norm(), 0, 0;
    double safety = std::pow(std::max(RR_(0) - 1.0, 0.0), 2);
    safety += 20*std::pow(std::max(RR_(0) - 2.0, 0.0), 4);
    discountFactorForSatefy_ = 1 - (RR_(0) > 2.5)*(RR_(0) - 2.5)/0.5; 
    rewards->record("c_safety", safety);
    desiredReward(rewards);

    // how much move along to the (past gc to current opponent) line
    Eigen::Vector2d pastToOppo = (targetPos_ - myHistory_[5].head(2)).normalized();
    Eigen::Vector2d pastToCur  = gc_.head(2) - myHistory_[5].head(2);
    double relative = pastToCur.dot(pastToOppo);
    rewards->record("e_relative", relative*(1 + (relative > 0)*(discountFactorForSatefy_ - 1)));
    const double r = std::max((gc_.head(2) - targetPos_).norm(), 0.0);
    rewards->record("h_distance", square(r));
    xyBefore_ << gc_.head(2), targetPos_;

    if((gc_.head(2) - targetPos_).norm() < 0.35){
      while((gc_.head(2) - targetPos_).norm() < 0.3){
        const double targetR = 1.3*(1+uniDist_(gen_));
        const double targetTheta = uniDist_(gen_)*2*M_PI;
        targetPos_ << targetR*cos(targetTheta), targetR*sin(targetTheta);
      }
      rewards->record("i_rpy", 3.0);
    }
    else{
      rewards->record("i_rpy", 0.0);
    }
    // max its r, p, z nominal, but its velocities are large
    Eigen::Vector2d rp;
    rp << quatToRPY(gc_.segment(3, 4)).head(2);
    double E = gc_(2) + 0.05*gv_(2)*gv_(2);
    rewards->record("j_zvz", -0.5*rp.squaredNorm() + 0.01*gv_.segment(2, 3).squaredNorm() + 10*E*E); // 17 -> 10
  }

  inline const Eigen::VectorXd &getObservation() {
    return obDouble_;
  }

  void setName(const std::string &name) {
    name_ = name;
  }

  void setOpponentName(const std::string &name) {
    opponentName_ = name;
  }

  void setPlayerNum(const int &playerNum) {
    playerNum_ = playerNum;
  }

  inline bool isTerminalState(raisim::World *world) {
    for (auto &contact: anymal_->getContacts()) {
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }
    }
    return false;
  }

  inline int getObDim() {
    return obDim_;
  }

  inline int getActionDim() {
    return actionDim_;
  }


 private:
  std::string name_, opponentName_;
  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;
  raisim::ArticulatedSystem *anymal_, *anymalO_;
  raisim::World *world_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int obDim_ = 0, actionDim_ = 0;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;
  // maintained variables for reward calculation
  int timer_;
  Eigen::Matrix3d rot_;
  Eigen::Vector4d xyBefore_, RR_;
  Eigen::VectorXd contactInfo_;
  double discountFactorForSatefy_;
  Eigen::Vector2d targetPos_;
  double targetSpeed_;
  // history related
  std::deque<Eigen::VectorXf> opponentHistory_;
  std::deque<Eigen::VectorXf> actionHistory_;
  std::deque<Eigen::VectorXd> myHistory_;

  
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::AnymalController_20002000::gen_;
thread_local std::normal_distribution<double> raisim::AnymalController_20002000::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::AnymalController_20002000::uniDist_(0., 1.);
}