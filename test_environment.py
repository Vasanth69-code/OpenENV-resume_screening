import unittest
from server.environment import ResumeScreeningEnvironment
from models import ScreeningAction

class TestResumeScreeningEnv(unittest.TestCase):

    def setUp(self):
        self.env = ResumeScreeningEnvironment()

    def test_easy_task_perfect_run(self):
        """Test that correctly evaluating all candidates yields exactly 1.0 reward."""
        obs = self.env.reset(task="easy")
        
        # In EASY task, target is: Reject, Select, Reject
        res1 = self.env.step(ScreeningAction(decision="reject", reasoning="no python"))
        res2 = self.env.step(ScreeningAction(decision="select", reasoning="has python"))
        res3 = self.env.step(ScreeningAction(decision="reject", reasoning="no python"))
        
        self.assertTrue(res3.done)
        total_reward = (res1.reward or 0) + (res2.reward or 0) + (res3.reward or 0)
        self.assertAlmostEqual(total_reward, 0.99, places=3)

    def test_medium_task_wrong_decisions(self):
        """Test that incorrect decisions yield 0.0 reward."""
        obs = self.env.reset(task="medium")
        
        # Making the opposite of correct decisions for everyone
        # Medium target: Select, Reject, Reject, Reject, Select
        res1 = self.env.step(ScreeningAction(decision="reject", reasoning="wrong"))
        res2 = self.env.step(ScreeningAction(decision="select", reasoning="wrong"))
        res3 = self.env.step(ScreeningAction(decision="select", reasoning="wrong"))
        res4 = self.env.step(ScreeningAction(decision="select", reasoning="wrong"))
        res5 = self.env.step(ScreeningAction(decision="reject", reasoning="wrong"))
        
        self.assertTrue(res5.done)
        total_reward = (res1.reward or 0) + (res2.reward or 0) + (res3.reward or 0) + (res4.reward or 0) + (res5.reward or 0)
        self.assertAlmostEqual(total_reward, 0.01, places=3)

    def test_hard_task_state_attributes(self):
        """Test state tracking and observation accuracy"""
        obs = self.env.reset(task="hard")
        
        self.assertEqual(obs.candidates_remaining, 5)
        self.assertEqual(obs.task_name, "hard")
        
        step_res = self.env.step(ScreeningAction(decision="reject", reasoning="test"))
        self.assertEqual(step_res.candidates_remaining, 4)

if __name__ == '__main__':
    unittest.main()
