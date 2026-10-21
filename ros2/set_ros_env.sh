#!/usr/bin/env sh
# Same DDS settings as the `humble` env of pixi_franka_crisp_ws, so this env sees the
# ZED topics. Every value can be overridden by exporting it before activation.

# The pip torch wheels otherwise pull in the system libstdc++, which is too old for the
# conda-forge OpenCV/ROS libraries loaded after it.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-5}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# The robot graph discovers over unicast only (see the comment in that file), so
# without this profile the node never finds the camera. Read in place, not copied.
FFS_FASTDDS_PROFILE="${FFS_FASTDDS_PROFILE:-${PIXI_PROJECT_ROOT}/../pixi_franka_crisp_ws/config/fastdds_config.xml}"
if [ "$RMW_IMPLEMENTATION" = "rmw_fastrtps_cpp" ]; then
    if [ -f "$FFS_FASTDDS_PROFILE" ]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="$FFS_FASTDDS_PROFILE"
    else
        echo "WARNING: Fast DDS profile $FFS_FASTDDS_PROFILE not found, using defaults." >&2
    fi
fi
