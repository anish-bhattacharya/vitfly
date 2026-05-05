# MAMBA BRANCH PERFORMANCE COMPARISON
====================================

## Test Status: INCOMPLETE - ROS Master Initialization Failure

All automated simulation tests failed due to ROS master communication errors when run via timeout command. The launch script requires interactive ROS environment that cannot be properly initialized in background/timeout mode.

**Error encountered**: `ERROR: Unable to communicate with master!` - ROS master fails to start when script is run via timeout, preventing all automated tests from executing.

## Known Working Configuration (Manual Test)

Branch B (MambaVisionSSM) was manually verified working with:
- Forward velocity: ~5.0 m/s
- Model inference: ~23ms/frame
- ROS topics publishing correctly
- RViz depth display working
- Model size: 12MB

## Branch Specifications

| Branch | Model Type | Model Size | Architecture |
|--------|-----------|------------|--------------|
| A | VMambaLSTM | 2.7MB | Vision Mamba + LSTM temporal fusion |
| B | MambaVisionSSM | 12MB | MambaVision encoder + SSM (stem_dim=48, stages=[64,128,192]) |
| C | CNNMamba3 | 6.8MB | CNN encoder + Mamba3 SSM |
| D | STHMamba | 1.5MB | Spatial-Temporal Hierarchical Mamba |
| E | DecisionMamba | 5.2MB | Decision-focused Mamba architecture |

## Test Configuration Applied

All tests were configured with:
- Goal distance: 20m (reduced from 60m)
- Timeout: 120s (increased from 40s)
- Position logging: Enabled
- Velocity calculation: Fixed (forward velocity now correct)
- Network config: Dynamic IP detection working
- RViz rendering: render:=True (depth display fixed)

## Analysis by Branch

### Branch A (VMambaLSTM)
**Status**: Test failed - ROS master error
**Expected strengths**: Smallest model (2.7MB), LSTM temporal fusion should provide good sequence understanding
**Expected challenges**: Limited capacity may struggle with complex visual scenes

### Branch B (MambaVisionSSM)
**Status**: Manually verified working
**Strengths**: Proven working in manual test, good velocity (~5.0 m/s), fast inference (23ms), robust MambaVision encoder
**Challenges**: Largest model (12MB), higher memory footprint
**Recommendation**: Currently the only verified working branch

### Branch C (CNNMamba3)
**Status**: Test failed - ROS master error
**Expected strengths**: CNN encoder provides strong visual features, medium size (6.8MB) balances capacity and efficiency
**Expected challenges**: CNN may add latency compared to pure Mamba architectures

### Branch D (STHMamba)
**Status**: Test failed - ROS master error
**Expected strengths**: Smallest model (1.5MB), spatial-temporal hierarchy should be efficient
**Expected challenges**: Very limited capacity may struggle with complex navigation

### Branch E (DecisionMamba)
**Status**: Test failed - ROS master error
**Expected strengths**: Decision-focused design should optimize for control outputs, medium size (5.2MB)
**Expected challenges**: May sacrifice visual understanding for decision quality

## RECOMMENDATION

**Use Branch B (MambaVisionSSM) for production** based on:

1. **Verified working**: Only branch with confirmed successful manual test
2. **Good performance**: 5.0 m/s forward velocity, 23ms inference time
3. **Robust architecture**: MambaVision encoder + SSM provides strong visual processing
4. **All fixes applied**: Network config, velocity calculation, RViz rendering all working

**Trade-offs accepted**:
- Largest model size (12MB) - acceptable for drone hardware
- Higher memory footprint - not a blocker for current platform

## ISSUES ENCOUNTERED

### Critical Issue: ROS Master Initialization Failure

**Problem**: When launch script is run via `timeout` command or in background mode, ROS master fails to initialize:
```
ERROR: Unable to communicate with master!
rospy.exceptions.ROSInitException: Failed to initialize time
```

**Root cause**: The launch script requires interactive ROS environment. When run via timeout/background:
1. ROS master (roscore) starts but cannot properly initialize
2. Evaluation node cannot communicate with master
3. Pilot node fails with "Failed to initialize time"
4. All tests timeout without collecting data

**Attempted solutions**:
- Background execution with `&` - Failed
- Foreground execution with timeout - Failed
- Sequential test execution - Failed

**Working solution**: Manual interactive execution (as verified for Branch B)

**Impact**: Cannot run automated batch testing of all 5 branches. Each branch must be tested manually and interactively.

### Recommendation for Future Testing

To properly compare all 5 branches, tests must be run:
1. **Manually and interactively** (not via timeout or background)
2. **One at a time** with full cleanup between tests
3. **With human observation** to verify ROS master startup
4. **Collecting logs manually** after each test completes

Alternatively, modify the launch script to:
- Pre-start roscore and wait for full initialization
- Add retry logic for ROS master connection
- Improve error handling for timeout scenarios

## Next Steps

1. **Production deployment**: Use Branch B (MambaVisionSSM) - verified working
2. **Future testing**: Run manual interactive tests for branches A, C, D, E to complete comparison
3. **Script improvement**: Fix launch script to support automated batch testing
4. **Monitoring**: Track Branch B performance in production to establish baseline metrics

---

**Test Date**: 2026-04-29
**Test Environment**: WSL2, ROS Noetic, Flightmare Unity simulator
**Configuration**: evaluation.yaml (goal=20m, timeout=120s)
