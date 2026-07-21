#!/usr/bin/env python3
"""
Demo script showing how to use the versioned vLLM defaults system.

This demonstrates:
1. Generating version-specific defaults
2. Managing multiple vLLM versions
3. Loading configurations with version-specific defaults
4. Comparing defaults across versions
"""

from auto_tune_vllm.core.config import StudyConfig
from auto_tune_vllm.utils import VLLMCLIParser, VLLMVersionManager


def demo_version_management():
    """Demonstrate version management capabilities."""
    print("🗂️  Version Management Demo")
    print("=" * 40)

    # Initialize version manager
    manager = VLLMVersionManager()

    # Show available versions
    print("📚 Available vLLM defaults versions:")
    print(manager.get_version_info_summary())

    # Get current vLLM version
    parser = VLLMCLIParser(venv_path=".venv")
    current_version = parser.get_vllm_version()
    print(f"🔍 Current vLLM version: {current_version}")

    # Load defaults for specific version
    try:
        defaults = manager.load_defaults(current_version)
        print(f"✅ Loaded defaults for version {current_version}")

        # Show some sample defaults
        print(f"\n📊 Sample defaults from v{current_version}:")
        for section, section_defaults in list(defaults.get("defaults", {}).items())[:2]:
            print(f"   {section}:")
            for param, value in list(section_defaults.items())[:3]:
                print(f"     {param} = {value}")
            if len(section_defaults) > 3:
                print(f"     ... and {len(section_defaults) - 3} more")
    except Exception as e:
        print(f"❌ Could not load defaults: {e}")


def demo_configuration_loading():
    """Demonstrate loading configurations with versioned defaults."""
    print("\n⚙️  Configuration Loading Demo")
    print("=" * 40)

    config_path = "examples/study_config.yaml"

    # Method 1: Load with auto-detected vLLM defaults
    print("📋 Method 1: Load with auto-detected vLLM defaults")
    try:
        config = StudyConfig.from_file(config_path)
        print("✅ Loaded config with auto-detected vLLM defaults")
        print(f"   Parameters: {len(config.parameters)}")

        # Show a parameter that uses vLLM defaults
        gpu_param = config.parameters.get("gpu_memory_utilization")
        if gpu_param:
            print(
                f"   gpu_memory_utilization: {gpu_param.min_value} - "
                f"{gpu_param.max_value}"
            )
            print("   (Uses vLLM default 0.9 as baseline)")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")

    # Method 2: Load with an explicit vLLM version (when defaults exist for that release)
    print("\n📋 Method 2: Load with explicit vLLM version (optional)")
    manager = VLLMVersionManager()
    versions = manager.list_available_versions()
    if not versions:
        print(
            "⚠️  No versioned defaults on disk — run scripts/generate_vllm_defaults.py first"
        )
        return
    version = versions[0].version
    try:
        config = StudyConfig.from_file(config_path, vllm_version=version)
        print(f"✅ Loaded config with vLLM v{version} defaults")
        print(f"   Parameters: {len(config.parameters)}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")


def demo_version_generation():
    """Demonstrate generating new version defaults."""
    print("\n🔧 Version Generation Demo")
    print("=" * 40)

    print("💡 To generate defaults for current vLLM version:")
    print("   python scripts/generate_vllm_defaults.py --verbose")
    print()
    print("💡 To generate defaults in a custom location:")
    print(
        "   python scripts/generate_vllm_defaults.py --output "
        "/path/to/defaults --verbose"
    )
    print()
    print("💡 To generate a single (non-versioned) defaults file:")
    print(
        "   python scripts/generate_vllm_defaults.py --no-versioned "
        "--output defaults.yaml"
    )


def demo_version_comparison():
    """Demonstrate comparing versions (when multiple exist)."""
    print("\n🔄 Version Comparison Demo")
    print("=" * 40)

    manager = VLLMVersionManager()
    versions = manager.list_available_versions()

    if len(versions) < 2:
        print("⚠️  Need at least 2 versions for comparison")
        print("   Generate defaults for different vLLM versions to compare them")
        print()
        print("💡 Example workflow:")
        print("   1. Install vLLM v0.10.0")
        print("   2. python scripts/generate_vllm_defaults.py")
        print("   3. Upgrade to vLLM v0.10.1")
        print("   4. python scripts/generate_vllm_defaults.py")
        print("   5. Compare versions using VLLMVersionManager.compare_versions()")
    else:
        # Show comparison example
        v1 = versions[0].version
        v2 = versions[1].version

        print(f"🔍 Comparing {v1} vs {v2}:")
        comparison = manager.compare_versions(v1, v2)

        print(f"   Added parameters: {comparison['summary']['added_count']}")
        print(f"   Removed parameters: {comparison['summary']['removed_count']}")
        print(f"   Changed parameters: {comparison['summary']['changed_count']}")
        print(f"   Unchanged parameters: {comparison['summary']['unchanged_count']}")


def main():
    """Run the versioned defaults demo."""
    print("🚀 Versioned vLLM Defaults Demo")
    print("=" * 50)

    try:
        demo_version_management()
        demo_configuration_loading()
        demo_version_generation()
        demo_version_comparison()

        print("\n✨ Demo completed successfully!")
        print("\n🎯 Key Benefits:")
        print("   ✅ No hardcoded defaults - automatically extracted from vLLM")
        print("   ✅ Version tracking - maintain defaults for each vLLM version")
        print("   ✅ Easy updates - regenerate defaults when vLLM updates")
        print("   ✅ Flexible loading - specify version or use latest")
        print("   ✅ Version comparison - track changes between vLLM versions")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
