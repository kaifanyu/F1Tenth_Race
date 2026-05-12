// pure_pursuit_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>

class PurePursuit : public rclcpp::Node {
public:
    PurePursuit() : Node("pure_pursuit_node") {
        // Core params
        declare_parameter("waypoint_file", "");
        declare_parameter("lookahead_distance", 0.8);
        declare_parameter("lookahead_gain", 0.5);
        declare_parameter("min_lookahead", 0.5);
        declare_parameter("max_lookahead", 2.0);
        declare_parameter("velocity", 0.8);
        declare_parameter("max_steering_angle", 0.4189);
        declare_parameter("wheelbase", 0.3302);
        declare_parameter("use_odom", true);
        declare_parameter("speed_lookahead", true);
        declare_parameter("min_speed_for_lookahead", 0.5);

        // Multi-lane params
        declare_parameter("scan_topic", "/scan");
        declare_parameter("scoring_horizon_m", 3.0);
        declare_parameter("safety_threshold_m", 0.35);
        declare_parameter("follow_distance_m", 1.5);
        declare_parameter("min_speed_scale", 0.3);

        waypoint_file_           = get_parameter("waypoint_file").as_string();
        lookahead_distance_      = get_parameter("lookahead_distance").as_double();
        lookahead_gain_          = get_parameter("lookahead_gain").as_double();
        min_lookahead_           = get_parameter("min_lookahead").as_double();
        max_lookahead_           = get_parameter("max_lookahead").as_double();
        velocity_                = get_parameter("velocity").as_double();
        max_steer_               = get_parameter("max_steering_angle").as_double();
        wheelbase_               = get_parameter("wheelbase").as_double();
        use_odom_                = get_parameter("use_odom").as_bool();
        speed_lookahead_         = get_parameter("speed_lookahead").as_bool();
        min_speed_for_lookahead_ = get_parameter("min_speed_for_lookahead").as_double();
        scoring_horizon_m_       = get_parameter("scoring_horizon_m").as_double();
        safety_threshold_m_      = get_parameter("safety_threshold_m").as_double();
        follow_distance_m_       = get_parameter("follow_distance_m").as_double();
        min_speed_scale_         = get_parameter("min_speed_scale").as_double();
        std::string scan_topic   = get_parameter("scan_topic").as_string();

        load_waypoints(waypoint_file_);

        // Start on middle lane.
        active_lane_ = 1;
        last_switch_time_ = now();

        // Publishers
        drive_pub_     = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        wp_viz_pub_    = create_publisher<visualization_msgs::msg::MarkerArray>("/waypoints_viz", 10);
        goal_viz_pub_  = create_publisher<visualization_msgs::msg::Marker>("/goal_waypoint_viz", 10);
        opp_viz_pub_   = create_publisher<visualization_msgs::msg::MarkerArray>("/opponents_viz", 10);

        // Subscribers
        if (use_odom_) {
            odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
                "/ego_racecar/odom", 10,
                std::bind(&PurePursuit::odom_callback, this, std::placeholders::_1));
            RCLCPP_INFO(get_logger(), "Subscribing to odom (simulator mode)");
        } else {
            pf_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
                "/pf/viz/inferred_pose", 10,
                std::bind(&PurePursuit::pose_callback, this, std::placeholders::_1));
            RCLCPP_INFO(get_logger(), "Subscribing to particle filter pose");
        }

        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            scan_topic, 10,
            std::bind(&PurePursuit::scan_callback, this, std::placeholders::_1));

        viz_timer_ = create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&PurePursuit::publish_waypoints_viz, this));
    }

private:
    struct Waypoint {
        double x;
        double y;
        double v;
    };

    struct Opponent {
        double x;
        double y;  // map frame
    };

    // Tunable constants (hardcoded per spec)
    static constexpr double SWITCH_COOLDOWN_S = 0.3;
    static constexpr double CLUSTER_RANGE_JUMP_M = 0.30;   // was 0.20 — more tolerant
    static constexpr int    CLUSTER_MIN_POINTS = 2;        // was 3 — accept smaller clusters
    static constexpr double CLUSTER_MAX_WIDTH_M = 0.60;    // was 0.40 — allow wider clusters
    static constexpr double OPP_ASSOC_RADIUS_M = 1.0;      // was 0.5 — track faster motion
    static constexpr int    CLUSTER_MAX_POINTS = 40;
    static constexpr double CLUSTER_MIN_WIDTH_M = 0.03;
    static constexpr double CLUSTER_MAX_RANGE_M = 6.0;
    static constexpr double LEFTWARD_PREFERENCE_W = 0.5;  // pull toward lane 0
    static constexpr double SWITCHING_PENALTY_W = 0.3;
    static constexpr double LANE_PREFERENCE_W = 0.7;   // pull toward racing line (lane 0)
    static constexpr double COLLISION_PENALTY_W = 100.0;  // dominates everything
    static constexpr int    OPP_PERSISTENCE_FRAMES = 2;
    
    std::array<std::vector<Waypoint>, 2> lanes_;  // lanes_[0]=left, [1]=middle, [2]=right
    size_t lane_size_ = 0;                         // common length

    // Params
    std::string waypoint_file_;
    double lookahead_distance_, lookahead_gain_, min_lookahead_, max_lookahead_;
    double velocity_, max_steer_, wheelbase_;
    bool use_odom_, speed_lookahead_;
    double min_speed_for_lookahead_;
    double scoring_horizon_m_, safety_threshold_m_, follow_distance_m_, min_speed_scale_;

    // State
    double current_speed_ = 0.0;
    double last_cmd_speed_ = 0.0;
    int active_lane_ = 0;
    rclcpp::Time last_switch_time_;

    // Latest pose (cached for scan_callback's frame transform)
    double pose_x_ = 0.0, pose_y_ = 0.0, pose_yaw_ = 0.0;
    bool have_pose_ = false;

    // Opponent tracking
    std::vector<Opponent> opponents_current_;          // confirmed opponents (after persistence)
    std::vector<Opponent> opponents_prev_detections_;  // raw last-frame detections

    // ROS handles
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr wp_viz_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_viz_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr opp_viz_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pf_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::TimerBase::SharedPtr viz_timer_;

    static double quaternion_to_yaw(double qx, double qy, double qz, double qw) {
        const double siny_cosp = 2.0 * (qw * qz + qx * qy);
        const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    void load_waypoints(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open waypoint file: %s", path.c_str());
            return;
        }

        std::string line;
        size_t parsed = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::stringstream ss(line);
            std::string tok;
            std::vector<double> vals;
            while (std::getline(ss, tok, ',')) {
                try { vals.push_back(std::stod(tok)); }
                catch (...) { vals.clear(); break; }
            }

            // Expect: x, y, v, lane_id
            if (vals.size() < 4) continue;

            int lane_id = static_cast<int>(vals[3]);
            if (lane_id < 0 || lane_id > 1) continue;

            lanes_[lane_id].push_back({vals[0], vals[1], vals[2]});
            ++parsed;
        }

        if (lanes_[0].empty() || lanes_[1].empty()) {
            RCLCPP_ERROR(get_logger(), "One or more lanes are empty after loading.");
            return;
        }

        if (lanes_[0].size() != lanes_[1].size()) {
            RCLCPP_WARN(get_logger(),
                "Lanes have different lengths (%zu, %zu). Truncating to minimum.",
                lanes_[0].size(), lanes_[1].size());
            const size_t m = std::min(lanes_[0].size(), lanes_[1].size());
            for (auto& L : lanes_) L.resize(m);
        }

        lane_size_ = lanes_[0].size();
        RCLCPP_INFO(get_logger(),
            "Loaded %zu waypoints across 2 lanes (%zu per lane)",
            parsed, lane_size_);
    }

    // ---------- Adaptive lookahead (operates on active lane) ----------
    double compute_reference_speed(size_t nearest_idx) const {
        if (lane_size_ == 0) return velocity_;
        if (use_odom_) {
            return std::max(current_speed_, min_speed_for_lookahead_);
        }
        const double wp_speed = lanes_[active_lane_][nearest_idx].v;
        return std::max((wp_speed > 0.0 ? wp_speed : last_cmd_speed_), min_speed_for_lookahead_);
    }

    double compute_lookahead(size_t nearest_idx) const {
        if (!speed_lookahead_) return lookahead_distance_;
        const double ref_speed = compute_reference_speed(nearest_idx);
        return std::clamp(lookahead_gain_ * ref_speed, min_lookahead_, max_lookahead_);
    }

    // ---------- Pose callbacks ----------
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        const auto& p = msg->pose.pose;
        pose_yaw_ = quaternion_to_yaw(p.orientation.x, p.orientation.y,
                                      p.orientation.z, p.orientation.w);
        pose_x_ = p.position.x;
        pose_y_ = p.position.y;
        have_pose_ = true;

        const double vx = msg->twist.twist.linear.x;
        const double vy = msg->twist.twist.linear.y;
        current_speed_ = std::hypot(vx, vy);

        pursue();
    }

    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        const auto& p = msg->pose;
        pose_yaw_ = quaternion_to_yaw(p.orientation.x, p.orientation.y,
                                      p.orientation.z, p.orientation.w);
        pose_x_ = p.position.x;
        pose_y_ = p.position.y;
        have_pose_ = true;
        pursue();
    }

    // ---------- LiDAR clustering -> opponents ----------
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        if (!have_pose_) return;

        std::vector<Opponent> raw_detections;

        const size_t n = msg->ranges.size();
        if (n < 4) return;

        // Group consecutive points whose neighbor-to-neighbor range step is small.
        std::vector<size_t> cluster_idx;
        cluster_idx.reserve(50);

        auto flush_cluster = [&](){
            if (cluster_idx.size() < CLUSTER_MIN_POINTS ||
                cluster_idx.size() > CLUSTER_MAX_POINTS) {
                cluster_idx.clear();
                return;
            }
            // Endpoints in vehicle frame
            const size_t i0 = cluster_idx.front();
            const size_t i1 = cluster_idx.back();
            const double a0 = msg->angle_min + i0 * msg->angle_increment;
            const double a1 = msg->angle_min + i1 * msg->angle_increment;
            const double r0 = msg->ranges[i0];
            const double r1 = msg->ranges[i1];
            const double x0 = r0 * std::cos(a0), y0 = r0 * std::sin(a0);
            const double x1 = r1 * std::cos(a1), y1 = r1 * std::sin(a1);
            const double width = std::hypot(x1 - x0, y1 - y0);

            if (width < CLUSTER_MIN_WIDTH_M || width > CLUSTER_MAX_WIDTH_M) {
                cluster_idx.clear();
                return;
            }

            // Centroid in vehicle frame, then map frame.
            double cx_v = 0.0, cy_v = 0.0;
            int count = 0;
            for (size_t k : cluster_idx) {
                const double a = msg->angle_min + k * msg->angle_increment;
                const double r = msg->ranges[k];
                if (!std::isfinite(r) || r <= 0.0 || r > CLUSTER_MAX_RANGE_M) continue;
                cx_v += r * std::cos(a);
                cy_v += r * std::sin(a);
                ++count;
            }
            cluster_idx.clear();
            if (count == 0) return;
            cx_v /= count;
            cy_v /= count;

            // Vehicle -> map (assumes LiDAR ~ base_link; offset fudge can be added later)
            const double cs = std::cos(pose_yaw_), sn = std::sin(pose_yaw_);
            const double mx = pose_x_ + cs * cx_v - sn * cy_v;
            const double my = pose_y_ + sn * cx_v + cs * cy_v;
            raw_detections.push_back({mx, my});
        };

        for (size_t i = 0; i < n; ++i) {
            const double r = msg->ranges[i];
            if (!std::isfinite(r) || r <= msg->range_min || r > CLUSTER_MAX_RANGE_M) {
                flush_cluster();
                continue;
            }
            if (cluster_idx.empty()) {
                cluster_idx.push_back(i);
                continue;
            }
            const double r_prev = msg->ranges[cluster_idx.back()];
            if (std::isfinite(r_prev) && std::abs(r - r_prev) < CLUSTER_RANGE_JUMP_M) {
                cluster_idx.push_back(i);
            } else {
                flush_cluster();
                cluster_idx.push_back(i);
            }
        }
        flush_cluster();

        // Cross-frame persistence: confirm a detection only if a similar one was
        // present in the previous frame.
        std::vector<Opponent> confirmed;
        for (const auto& d : raw_detections) {
            for (const auto& prev : opponents_prev_detections_) {
                if (std::hypot(d.x - prev.x, d.y - prev.y) < OPP_ASSOC_RADIUS_M) {
                    confirmed.push_back(d);
                    break;
                }
            }
        }
        (void)OPP_PERSISTENCE_FRAMES;  // 2-frame persistence is what the above implements
        opponents_current_ = confirmed;
        opponents_prev_detections_ = raw_detections;

        publish_opponents_viz();
    }

    // ---------- Lane scoring ----------
    size_t nearest_idx_on_lane(int lane, double x, double y) const {
        size_t best = 0;
        double best_d = std::numeric_limits<double>::max();
        for (size_t i = 0; i < lane_size_; ++i) {
            const double d = std::hypot(lanes_[lane][i].x - x, lanes_[lane][i].y - y);
            if (d < best_d) { best_d = d; best = i; }
        }
        return best;
    }

    // Returns minimum opponent-to-waypoint distance over the forward horizon
    // for the given lane. Smaller = more dangerous. If no opponents, returns +inf.
    double min_opponent_clearance(int lane, size_t start_idx) const {
        if (opponents_current_.empty()) return std::numeric_limits<double>::infinity();

        double accum = 0.0;
        double min_d = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < lane_size_ && accum < scoring_horizon_m_; ++k) {
            const size_t i = (start_idx + k) % lane_size_;
            const size_t j = (start_idx + k + 1) % lane_size_;
            const auto& wp = lanes_[lane][i];
            for (const auto& opp : opponents_current_) {
                const double d = std::hypot(wp.x - opp.x, wp.y - opp.y);
                if (d < min_d) min_d = d;
            }
            accum += std::hypot(lanes_[lane][j].x - wp.x, lanes_[lane][j].y - wp.y);
        }
        return min_d;
    }

    int select_lane(size_t nearest_idx_active) {
        // Map nearest index from active lane to sibling indices (lanes are aligned).
        const size_t base_idx = nearest_idx_active;

        std::array<double, 2> cost{};
        std::array<bool, 2> blocked{};

        for (int lane = 0; lane < 2; ++lane) {
            const double clearance = min_opponent_clearance(lane, base_idx);

            double collision_cost = 0.0;
            if (clearance < safety_threshold_m_) {
                blocked[lane] = true;
                collision_cost = COLLISION_PENALTY_W;
            } else if (std::isfinite(clearance)) {
                const double margin = clearance - safety_threshold_m_;
                collision_cost = 5.0 * std::exp(-1.0 * margin);  // strong/slow-decay
            }

            // Racing-line preference: lane 0 (racing line) is preferred over lane 1 (overtake)
            const double lane_pref_cost = LANE_PREFERENCE_W * lane;

            // Switching penalty
            const double switch_cost = (lane != active_lane_) ? SWITCHING_PENALTY_W : 0.0;

            cost[lane] = collision_cost + lane_pref_cost + switch_cost;
        }

        // Cooldown: only allow a switch if enough time has passed
        const double since_switch = (now() - last_switch_time_).seconds();
        const bool can_switch = since_switch >= SWITCH_COOLDOWN_S;

        // Find best lane
        int best_lane = active_lane_;
        double best_cost = cost[active_lane_];
        for (int lane = 0; lane < 2; ++lane) {
            if (cost[lane] < best_cost) {
                best_cost = cost[lane];
                best_lane = lane;
            }
        }

        // If our current lane is blocked, override cooldown — safety first
        const bool current_blocked = blocked[active_lane_];

        if (best_lane != active_lane_ && (can_switch || current_blocked)) {
            RCLCPP_INFO(get_logger(),
                "Lane switch: %d -> %d (cost %.2f -> %.2f, current_blocked=%d)",
                active_lane_, best_lane, cost[active_lane_], best_cost, current_blocked ? 1 : 0);
            active_lane_ = best_lane;
            last_switch_time_ = now();
        }
        return active_lane_;
    }

    // ---------- Speed modulation when opponent ahead on chosen lane ----------
    double speed_scale_for_opponent(int lane, size_t start_idx) const {
        if (opponents_current_.empty()) return 1.0;

        // Find the closest opponent that's actually ahead of us on this lane,
        // using minimum distance to the lane's forward window.
        double accum = 0.0;
        double closest_ahead = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < lane_size_ && accum < follow_distance_m_; ++k) {
            const size_t i = (start_idx + k) % lane_size_;
            const size_t j = (start_idx + k + 1) % lane_size_;
            const auto& wp = lanes_[lane][i];
            for (const auto& opp : opponents_current_) {
                const double d = std::hypot(wp.x - opp.x, wp.y - opp.y);
                if (d < safety_threshold_m_ * 1.5 && accum < closest_ahead) {
                    closest_ahead = accum;
                }
            }
            accum += std::hypot(lanes_[lane][j].x - wp.x, lanes_[lane][j].y - wp.y);
        }

        if (!std::isfinite(closest_ahead)) return 1.0;

        // Linear ramp: 0 m -> min scale, follow_distance -> 1.0
        const double t = std::clamp(closest_ahead / follow_distance_m_, 0.0, 1.0);
        return min_speed_scale_ + (1.0 - min_speed_scale_) * t;
    }

    // ---------- Main control step ----------
    void pursue() {
        if (lane_size_ == 0) return;

        // Localize on current active lane
        size_t nearest = nearest_idx_on_lane(active_lane_, pose_x_, pose_y_);

        // Pick best lane (may update active_lane_)
        const int chosen = select_lane(nearest);

        // Re-localize on the (possibly new) chosen lane to be safe
        if (chosen != static_cast<int>(active_lane_)) {
            // shouldn't happen since select_lane updates active_lane_, but defensive
        }
        nearest = nearest_idx_on_lane(active_lane_, pose_x_, pose_y_);

        const auto& lane = lanes_[active_lane_];

        // Adaptive lookahead
        const double L = compute_lookahead(nearest);

        // Find goal: first waypoint beyond L, searching forward
        size_t goal_idx = nearest;
        for (size_t i = 0; i < lane_size_; ++i) {
            const size_t idx = (nearest + i) % lane_size_;
            const double d = std::hypot(lane[idx].x - pose_x_, lane[idx].y - pose_y_);
            if (d >= L) { goal_idx = idx; break; }
        }
        const auto& goal = lane[goal_idx];

        // Transform goal into vehicle frame
        const double dx = goal.x - pose_x_;
        const double dy = goal.y - pose_y_;
        const double gx_v =  dx * std::cos(pose_yaw_) + dy * std::sin(pose_yaw_);
        const double gy_v = -dx * std::sin(pose_yaw_) + dy * std::cos(pose_yaw_);

        const double L_act = std::hypot(gx_v, gy_v);
        if (L_act < 1e-3) return;

        const double curvature = 2.0 * gy_v / (L_act * L_act);
        double steer = std::atan(curvature * wheelbase_);
        steer = std::clamp(steer, -max_steer_, max_steer_);

        // Speed: waypoint-prescribed * opponent slowdown
        double speed = goal.v;
        const double scale = speed_scale_for_opponent(active_lane_, nearest);
        speed *= scale;
        last_cmd_speed_ = speed;

        ackermann_msgs::msg::AckermannDriveStamped drive;
        drive.header.stamp = now();
        drive.header.frame_id = "base_link";
        drive.drive.steering_angle = static_cast<float>(steer);
        drive.drive.speed = static_cast<float>(speed);
        drive_pub_->publish(drive);

        publish_goal_viz(goal.x, goal.y);
    }

    // ---------- Visualization ----------
    void publish_waypoints_viz() {
        visualization_msgs::msg::MarkerArray arr;
        // Color per lane: lane 0 cyan, lane 1 yellow, lane 2 magenta. Active lane brighter.
        const std::array<std::array<float, 3>, 2> base_colors = {{
            {{0.0f, 1.0f, 0.0f}},   // lane 0: racing line — green
            {{1.0f, 0.5f, 0.0f}},   // lane 1: overtake — orange
        }};

        int marker_id = 0;
        for (int lane = 0; lane < 2; ++lane) {
            const bool is_active = (lane == active_lane_);
            for (size_t i = 0; i < lanes_[lane].size(); ++i) {
                visualization_msgs::msg::Marker m;
                m.header.frame_id = "map";
                m.header.stamp = now();
                m.ns = "waypoints";
                m.id = marker_id++;
                m.type = visualization_msgs::msg::Marker::SPHERE;
                m.action = visualization_msgs::msg::Marker::ADD;
                m.pose.position.x = lanes_[lane][i].x;
                m.pose.position.y = lanes_[lane][i].y;
                m.pose.position.z = 0.0;
                m.pose.orientation.w = 1.0;
                const double s = is_active ? 0.12 : 0.07;
                m.scale.x = m.scale.y = m.scale.z = s;
                m.color.a = is_active ? 1.0f : 0.5f;
                m.color.r = base_colors[lane][0];
                m.color.g = base_colors[lane][1];
                m.color.b = base_colors[lane][2];
                arr.markers.push_back(m);
            }
        }
        wp_viz_pub_->publish(arr);
    }

    void publish_goal_viz(double gx, double gy) {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = now();
        m.ns = "goal";
        m.id = 0;
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = gx;
        m.pose.position.y = gy;
        m.pose.position.z = 0.0;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.25;
        m.color.a = 1.0f;
        m.color.r = 1.0f;
        goal_viz_pub_->publish(m);
    }

    void publish_opponents_viz() {
        visualization_msgs::msg::MarkerArray arr;
        // Clear previous
        visualization_msgs::msg::Marker clr;
        clr.header.frame_id = "map";
        clr.header.stamp = now();
        clr.ns = "opponents";
        clr.action = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(clr);

        int id = 0;
        for (const auto& opp : opponents_current_) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map";
            m.header.stamp = now();
            m.ns = "opponents";
            m.id = id++;
            m.type = visualization_msgs::msg::Marker::CUBE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = opp.x;
            m.pose.position.y = opp.y;
            m.pose.position.z = 0.2;
            m.pose.orientation.w = 1.0;
            m.scale.x = m.scale.y = 0.12;
            m.scale.z = 0.20;
            m.color.a = 0.9f;
            m.color.r = 1.0f;
            m.color.g = 0.3f;
            m.color.b = 0.0f;
            arr.markers.push_back(m);
        }
        opp_viz_pub_->publish(arr);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}