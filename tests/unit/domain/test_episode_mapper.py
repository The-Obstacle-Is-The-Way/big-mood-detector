"""
Test Episode Mapper

Tests for the EpisodeMapper utility that handles episode type conversions.
"""

class TestEpisodeMapper:
    """Test episode type mapping utility."""

    def test_to_dsm5_episode_type_depressive(self):
        """Test mapping depressive_episode to DSM-5 format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_dsm5_episode_type("depressive_episode") == "depressive"

    def test_to_dsm5_episode_type_manic(self):
        """Test mapping manic_episode to DSM-5 format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_dsm5_episode_type("manic_episode") == "manic"

    def test_to_dsm5_episode_type_hypomanic(self):
        """Test mapping hypomanic_episode to DSM-5 format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_dsm5_episode_type("hypomanic_episode") == "hypomanic"

    def test_to_dsm5_episode_type_mixed(self):
        """Test mapping mixed_episode to DSM-5 format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_dsm5_episode_type("mixed_episode") == "mixed"

    def test_to_dsm5_episode_type_euthymic(self):
        """Test mapping euthymic (stable) to DSM-5 format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_dsm5_episode_type("euthymic") == "euthymic"

    def test_to_dsm5_episode_type_unknown(self):
        """Test unknown episode type returns as-is."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_dsm5_episode_type("unknown_type") == "unknown_type"

    def test_to_treatment_episode_type_depressive(self):
        """Test mapping depressive_episode to treatment format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert (
            EpisodeMapper.to_treatment_episode_type("depressive_episode")
            == "depressive"
        )

    def test_to_treatment_episode_type_manic(self):
        """Test mapping manic_episode to treatment format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_treatment_episode_type("manic_episode") == "manic"

    def test_to_treatment_episode_type_hypomanic(self):
        """Test mapping hypomanic_episode to treatment format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert (
            EpisodeMapper.to_treatment_episode_type("hypomanic_episode") == "hypomanic"
        )

    def test_to_treatment_episode_type_mixed(self):
        """Test mapping mixed_episode to treatment format."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_treatment_episode_type("mixed_episode") == "mixed"

    def test_to_treatment_episode_type_with_features(self):
        """Test mapping episodes with mixed features."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert (
            EpisodeMapper.to_treatment_episode_type("depressive_with_mixed_features")
            == "depressive_mixed"
        )
        assert (
            EpisodeMapper.to_treatment_episode_type("manic_with_mixed_features")
            == "manic_mixed"
        )

    def test_bidirectional_mapping(self):
        """Test that mappings can go both ways."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        # From clinical diagnosis to DSM-5
        assert EpisodeMapper.to_dsm5_episode_type("depressive_episode") == "depressive"

        # From DSM-5 back to clinical diagnosis
        assert (
            EpisodeMapper.from_dsm5_episode_type("depressive") == "depressive_episode"
        )

        # Round trip
        diagnosis = "manic_episode"
        dsm5 = EpisodeMapper.to_dsm5_episode_type(diagnosis)
        back = EpisodeMapper.from_dsm5_episode_type(dsm5)
        assert back == diagnosis

    def test_case_insensitive_mapping(self):
        """Test that mappings handle case variations."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.to_dsm5_episode_type("DEPRESSIVE_EPISODE") == "depressive"
        assert EpisodeMapper.to_dsm5_episode_type("Manic_Episode") == "manic"

    def test_get_all_episode_types(self):
        """Test getting all known episode types."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        all_types = EpisodeMapper.get_all_episode_types()

        # Should include at least the main types
        assert "depressive_episode" in all_types
        assert "manic_episode" in all_types
        assert "hypomanic_episode" in all_types
        assert "mixed_episode" in all_types
        assert "euthymic" in all_types

    def test_is_valid_episode_type(self):
        """Test validation of episode types."""
        from big_mood_detector.domain.utils.episode_mapper import EpisodeMapper

        assert EpisodeMapper.is_valid_episode_type("depressive_episode") is True
        assert EpisodeMapper.is_valid_episode_type("manic_episode") is True
        assert EpisodeMapper.is_valid_episode_type("invalid_episode") is False
        assert EpisodeMapper.is_valid_episode_type("") is False
        assert EpisodeMapper.is_valid_episode_type(None) is False
