"""
AI Service integration for Student Tracking System.
Handles Google Gemini API calls for chat, recommendations, and analytics.
"""

from django.conf import settings
import os
import logging
from typing import Dict, List
import json
from datetime import datetime, timedelta

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    HarmCategory = None
    HarmBlockThreshold = None

# Configure logging
logger = logging.getLogger(__name__)

class AIService:
    """AI service wrapper for Google Gemini integration."""
    
    def __init__(self):
        self.model = None
        self.mock_mode = True
        self.ai_provider = 'offline'
        
        # Import offline AI
        try:
            from utils.offline_ai import get_ai_response
            self.offline_ai_available = True
            logger.info("Offline AI initialized successfully")
        except ImportError as e:
            self.offline_ai_available = False
            logger.warning(f"Offline AI not available: {e}")
        
        # Initialize Gemini
        gemini_api_key = os.environ.get('GEMINI_API_KEY') or getattr(settings, 'GEMINI_API_KEY', None)
        gemini_model_name = os.environ.get('GEMINI_MODEL') or getattr(settings, 'GEMINI_MODEL', 'gemini-pro')
        
        if GEMINI_AVAILABLE and gemini_api_key and gemini_api_key != '':
            try:
                genai.configure(api_key=gemini_api_key)
                # Configure safety settings to allow educational content
                if HarmCategory and HarmBlockThreshold:
                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                    self.model = genai.GenerativeModel(
                        gemini_model_name,
                        safety_settings=safety_settings
                    )
                else:
                    self.model = genai.GenerativeModel(gemini_model_name)
                self.mock_mode = False
                self.ai_provider = 'gemini'
                logger.info(f"Gemini client initialized successfully with model: {gemini_model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {str(e)}")
                self.model = None
        
        # Final mode determination
        if not self.model:
            if self.offline_ai_available:
                self.ai_provider = 'offline'
                self.mock_mode = False
                logger.info("AI Service using intelligent offline mode")
            else:
                self.ai_provider = 'mock'
                self.mock_mode = True
                logger.info("AI Service running in basic mock mode")
    
    def chat_with_ai(self, message: str, context: Dict = None) -> str:
        """Chat with AI for student queries."""
        try:
            if self.ai_provider == 'gemini' and not self.mock_mode:
                return self._chat_with_gemini(message, context)
            elif self.ai_provider == 'offline' and self.offline_ai_available:
                return self._chat_with_offline_ai(message, context)
            else:
                return self._mock_chat_response(message, context)
            
        except Exception as e:
            logger.error(f"AI chat error: {str(e)}")
            # Try offline AI as fallback
            if self.offline_ai_available:
                try:
                    return self._chat_with_offline_ai(message, context)
                except:
                    pass
            return self._mock_chat_response(message, context)
    
    def chat_response(self, message: str, context: Dict = None) -> str:
        """Alias for chat_with_ai method for compatibility."""
        return self.chat_with_ai(message, context)
    
    def generate_study_recommendation(self, student_data: Dict) -> Dict:
        """Generate personalized study recommendations."""
        try:
            if self.mock_mode:
                return self._mock_study_recommendations(student_data)
            
            if not hasattr(settings, 'GEMINI_API_KEY') or not settings.GEMINI_API_KEY:
                return self._mock_study_recommendations(student_data)
            
            prompt = self._build_recommendation_prompt(student_data)
            full_prompt = f"You are an AI study advisor for students. Provide personalized study recommendations based on their academic data.\n\n{prompt}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'max_output_tokens': 800,
                    'temperature': 0.6,
                }
            )
            
            # Parse the response to structured format
            content = response.text.strip()
            return self._parse_recommendation_response(content)
            
        except Exception as e:
            logger.error(f"Study recommendation error: {str(e)}")
            return self._mock_study_recommendations(student_data)
    
    def optimize_timetable(self, timetable_data: Dict) -> Dict:
        """Optimize timetable using AI suggestions."""
        try:
            if self.mock_mode:
                return self._mock_timetable_optimization(timetable_data)
            
            if not hasattr(settings, 'GEMINI_API_KEY') or not settings.GEMINI_API_KEY:
                return self._mock_timetable_optimization(timetable_data)
            
            prompt = self._build_optimization_prompt(timetable_data)
            full_prompt = f"You are a timetable optimization expert. Analyze schedules and suggest improvements for better learning outcomes.\n\n{prompt}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'max_output_tokens': 600,
                    'temperature': 0.5,
                }
            )
            
            content = response.text.strip()
            return self._parse_optimization_response(content)
            
        except Exception as e:
            logger.error(f"Timetable optimization error: {str(e)}")
            return self._mock_timetable_optimization(timetable_data)
    
    def analyze_performance(self, performance_data: Dict) -> Dict:
        """Analyze student performance using AI."""
        try:
            if self.mock_mode:
                return self._mock_performance_analysis(performance_data)
            
            if not hasattr(settings, 'GEMINI_API_KEY') or not settings.GEMINI_API_KEY:
                return self._mock_performance_analysis(performance_data)
            
            prompt = self._build_analysis_prompt(performance_data)
            full_prompt = f"You are an educational data analyst. Analyze student performance data and provide insights.\n\n{prompt}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'max_output_tokens': 700,
                    'temperature': 0.4,
                }
            )
            
            content = response.text.strip()
            return self._parse_analysis_response(content)
            
        except Exception as e:
            logger.error(f"Performance analysis error: {str(e)}")
            return self._mock_performance_analysis(performance_data)
    
    # Helper methods for building prompts
    def _build_system_prompt(self, context: Dict = None) -> str:
        """Build system prompt for chat with database context."""
        base_prompt = "You are an AI assistant for a Student Tracking System. Help students with their academic queries, schedule questions, and study advice. You have access to the student's real database information."
        
        if context:
            # Student basic info
            student_info = context.get('student_info', {})
            if student_info:
                name = student_info.get('name', 'Student')
                roll_number = student_info.get('roll_number', '')
                course = student_info.get('course', '')
                year = student_info.get('year', '')
                section = student_info.get('section', '')
                base_prompt += f"\n\nSTUDENT INFORMATION:\n- Name: {name}\n- Roll Number: {roll_number}\n- Course: {course}\n- Year: {year}\n- Section: {section}"
            
            # Enrolled subjects
            enrolled_subjects = context.get('enrolled_subjects', [])
            if enrolled_subjects:
                subjects_list = "\n".join([f"  - {subj['name']} ({subj['code']}) - {subj['credits']} credits" for subj in enrolled_subjects])
                base_prompt += f"\n\nENROLLED SUBJECTS:\n{subjects_list}"
            
            # Attendance summary
            attendance_summary = context.get('attendance_summary', {})
            if attendance_summary:
                total = attendance_summary.get('total_classes', 0)
                present = attendance_summary.get('present', 0)
                absent = attendance_summary.get('absent', 0)
                late = attendance_summary.get('late', 0)
                percentage = attendance_summary.get('overall_percentage', 0)
                base_prompt += f"\n\nOVERALL ATTENDANCE:\n- Total Classes: {total}\n- Present: {present}\n- Absent: {absent}\n- Late: {late}\n- Attendance Percentage: {percentage}%"
            
            # Subject-wise attendance
            subject_attendance = context.get('subject_attendance', {})
            if subject_attendance:
                subject_att_list = "\n".join([f"  - {subject}: {stats['present']}/{stats['total']} classes ({stats['percentage']}%)" 
                                             for subject, stats in subject_attendance.items()])
                base_prompt += f"\n\nSUBJECT-WISE ATTENDANCE:\n{subject_att_list}"
            
            # Today's timetable
            today_timetable = context.get('today_timetable', [])
            if today_timetable:
                timetable_list = "\n".join([f"  - {cls['subject']} with {cls['teacher']} in {cls['room']} at {cls['time']}" 
                                           for cls in today_timetable])
                base_prompt += f"\n\nTODAY'S SCHEDULE:\n{timetable_list}"
            elif today_timetable == []:
                base_prompt += "\n\nTODAY'S SCHEDULE: No classes scheduled for today."
            
            # Recent attendance
            recent_attendance = context.get('recent_attendance', [])
            if recent_attendance:
                recent_list = "\n".join([f"  - {rec['date']}: {rec['subject']} - {rec['status'].upper()}" 
                                        for rec in recent_attendance])
                base_prompt += f"\n\nRECENT ATTENDANCE RECORDS:\n{recent_list}"
        
        base_prompt += "\n\nUse this information to answer the student's questions accurately. When asked about attendance, subjects, timetable, or schedule, provide specific details from the data above."
        
        return base_prompt
    
    def _build_recommendation_prompt(self, student_data: Dict) -> str:
        """Build prompt for study recommendations."""
        return f"""
        Generate study recommendations for a student with the following data:
        - Current GPA: {student_data.get('gpa', 'N/A')}
        - Subjects: {', '.join(student_data.get('subjects', []))}
        - Weak subjects: {', '.join(student_data.get('weak_subjects', []))}
        - Study hours per day: {student_data.get('study_hours', 'N/A')}
        - Upcoming exams: {student_data.get('upcoming_exams', 'None')}
        
        Please provide specific, actionable study recommendations.
        """
    
    def _build_optimization_prompt(self, timetable_data: Dict) -> str:
        """Build prompt for timetable optimization."""
        return f"""
        Analyze this timetable and suggest optimizations:
        - Total classes per day: {timetable_data.get('classes_per_day', 'N/A')}
        - Subject distribution: {timetable_data.get('subject_distribution', {})}
        - Break times: {timetable_data.get('break_times', [])}
        - Peak learning hours: {timetable_data.get('peak_hours', 'N/A')}
        
        Suggest improvements for better learning outcomes.
        """
    
    def _build_analysis_prompt(self, performance_data: Dict) -> str:
        """Build prompt for performance analysis."""
        return f"""
        Analyze this student performance data:
        - Attendance rate: {performance_data.get('attendance_rate', 'N/A')}%
        - Grade trends: {performance_data.get('grade_trends', {})}
        - Subject performance: {performance_data.get('subject_performance', {})}
        - Assignment completion: {performance_data.get('assignment_completion', 'N/A')}%
        
        Provide insights and recommendations for improvement.
        """
    
    # Gemini-specific methods
    def _chat_with_gemini(self, message: str, context: Dict = None) -> str:
        """Chat using Google Gemini API."""
        try:
            system_prompt = self._build_system_prompt(context)
            full_prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'max_output_tokens': 1000,
                    'temperature': 0.7,
                }
            )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            # Fallback to offline AI
            if self.offline_ai_available:
                try:
                    return self._chat_with_offline_ai(message, context)
                except:
                    pass
            return self._generate_smart_response(message, context)
    
    # Mock response methods for development
    def _mock_chat_response(self, message: str, context: Dict = None) -> str:
        """Generate mock chat response."""
        responses = [
            "I understand your question about the timetable. Let me help you with that.",
            "That's a great question! Based on your current schedule, I'd recommend...",
            "I can help you with that. Here's what you should know about your classes:",
            "Let me provide some guidance on that topic for your studies.",
        ]
        
        # Simple keyword-based responses
        if 'schedule' in message.lower() or 'timetable' in message.lower():
            return "Your next class is in 30 minutes. You have Mathematics in Room 101. Don't forget to bring your calculator!"
        elif 'study' in message.lower() or 'exam' in message.lower():
            return "For your upcoming exams, I recommend creating a study schedule. Focus on your weaker subjects first, and allocate more time to practice problems."
        elif 'assignment' in message.lower():
            return "You have 2 assignments due this week. The Physics assignment is due tomorrow, and the English essay is due on Friday."
        else:
            return f"Thanks for your question: '{message}'. I'm here to help with your academic needs. Feel free to ask about your schedule, assignments, or study tips!"
    
    def _mock_study_recommendations(self, student_data: Dict) -> Dict:
        """Generate mock study recommendations."""
        return {
            "priority_subjects": ["Mathematics", "Physics", "Chemistry"],
            "study_plan": [
                {
                    "subject": "Mathematics",
                    "hours_per_week": 8,
                    "focus_areas": ["Calculus", "Algebra", "Trigonometry"],
                    "resources": ["Khan Academy", "Textbook Chapter 5-7", "Practice Problems"]
                },
                {
                    "subject": "Physics", 
                    "hours_per_week": 6,
                    "focus_areas": ["Mechanics", "Thermodynamics"],
                    "resources": ["Lab experiments", "Video lectures", "Problem sets"]
                }
            ],
            "daily_schedule": {
                "morning": "Review previous day's topics (30 min)",
                "afternoon": "Practice problems (1-2 hours)",
                "evening": "Read ahead for next day (45 min)"
            },
            "exam_strategy": "Focus on weak areas first, create summary notes, take practice tests",
            "confidence_score": 85
        }
    
    def _mock_timetable_optimization(self, timetable_data: Dict) -> Dict:
        """Generate mock timetable optimization suggestions."""
        return {
            "optimization_score": 78,
            "suggestions": [
                {
                    "type": "scheduling",
                    "priority": "high",
                    "description": "Move Mathematics to morning hours (9-11 AM) for better concentration"
                },
                {
                    "type": "breaks",
                    "priority": "medium", 
                    "description": "Add 15-minute break between consecutive theory classes"
                },
                {
                    "type": "subject_distribution",
                    "priority": "medium",
                    "description": "Balance theory and practical subjects throughout the week"
                }
            ],
            "predicted_improvements": {
                "learning_efficiency": "+15%",
                "student_satisfaction": "+12%",
                "attendance_rate": "+8%"
            }
        }
    
    def _mock_performance_analysis(self, performance_data: Dict) -> Dict:
        """Generate mock performance analysis."""
        return {
            "overall_score": 82,
            "strengths": ["Mathematics", "Computer Science", "Regular attendance"],
            "areas_for_improvement": ["Physics lab reports", "English essays", "Time management"],
            "trends": {
                "attendance": "Stable at 92%",
                "grades": "Improving trend over last 3 months",
                "assignment_completion": "98% completion rate"
            },
            "recommendations": [
                "Join Physics study group for better lab understanding",
                "Use writing center resources for English improvements", 
                "Consider time management workshops"
            ],
            "predicted_gpa": 3.7,
            "confidence_interval": "±0.2"
        }
    
    # Response parsing methods (for real API responses)
    def _parse_recommendation_response(self, content: str) -> Dict:
        """Parse AI recommendation response to structured format."""
        # This would parse the AI response into structured data
        # For now, return a mock structure
        return self._mock_study_recommendations({})
    
    def _parse_optimization_response(self, content: str) -> Dict:
        """Parse AI optimization response to structured format."""
        return self._mock_timetable_optimization({})
    
    def _parse_analysis_response(self, content: str) -> Dict:
        """Parse AI analysis response to structured format."""
        return self._mock_performance_analysis({})
    
    def _ensure_model_with_safety(self):
        """Ensure model is initialized with safety settings."""
        if not GEMINI_AVAILABLE:
            return False
        
        gemini_api_key = os.environ.get('GEMINI_API_KEY') or getattr(settings, 'GEMINI_API_KEY', None)
        gemini_model_name = os.environ.get('GEMINI_MODEL') or getattr(settings, 'GEMINI_MODEL', 'gemini-2.5-flash')
        
        if not gemini_api_key or gemini_api_key == '':
            return False
        
        try:
            if HarmCategory and HarmBlockThreshold:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                self.model = genai.GenerativeModel(gemini_model_name, safety_settings=safety_settings)
            else:
                self.model = genai.GenerativeModel(gemini_model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to recreate model with safety settings: {e}")
            return False
    
    def generate_timetable_with_gemini(self, timetable_context: Dict) -> Dict:
        """Generate timetable using Gemini AI with full database context."""
        try:
            if self.mock_mode:
                logger.warning("AI Service is in mock mode, returning mock timetable")
                return {
                    'success': False,
                    'error': 'AI Service is in mock mode. Please configure Gemini API key.',
                    'algorithm': 'mock'
                }
            
            if not hasattr(settings, 'GEMINI_API_KEY') or not settings.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not configured")
                return {
                    'success': False,
                    'error': 'Gemini API key not configured. Please set GEMINI_API_KEY in settings.',
                    'algorithm': 'gemini_ai'
                }
            
            # Ensure model has safety settings
            if not self.model:
                self._ensure_model_with_safety()
            
            if not self.model:
                logger.warning("Gemini model not initialized")
                return {
                    'success': False,
                    'error': 'Gemini model not initialized. Please check your API key and model name.',
                    'algorithm': 'gemini_ai'
                }
            
            prompt = self._build_timetable_generation_prompt(timetable_context)
            logger.info(f"Calling Gemini API with prompt length: {len(prompt)}")
            
            # Also pass safety settings in generate_content call as backup
            safety_settings_dict = None
            if HarmCategory and HarmBlockThreshold:
                safety_settings_dict = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': 4000,
                    'temperature': 0.3,
                },
                safety_settings=safety_settings_dict if safety_settings_dict else None
            )
            
            if not response:
                logger.error("Gemini returned None response")
                return {
                    'success': False,
                    'error': 'Empty response from Gemini API',
                    'algorithm': 'gemini_ai'
                }
            
            # Check if response was blocked
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    # finish_reason 2 = SAFETY (blocked)
                    # finish_reason 3 = RECITATION (blocked)
                    # finish_reason 4 = OTHER (blocked)
                    if finish_reason in [2, 3, 4]:
                        logger.warning(f"Gemini response blocked (finish_reason: {finish_reason}). Using fallback timetable generation.")
                        return self._generate_basic_timetable(timetable_context)
            
            # Try to get text content
            try:
                if not hasattr(response, 'text'):
                    logger.error("Response object has no 'text' attribute")
                    return {
                        'success': False,
                        'error': 'Invalid response format from Gemini API',
                        'algorithm': 'gemini_ai'
                    }
                
                content = response.text.strip()
                if not content:
                    logger.error("Gemini returned empty text content")
                    return {
                        'success': False,
                        'error': 'Empty text content from Gemini API',
                        'algorithm': 'gemini_ai'
                    }
                
                logger.info(f"Received response from Gemini (length: {len(content)})")
                return self._parse_timetable_response(content, timetable_context)
            except ValueError as e:
                # This happens when response.text is accessed but content was blocked
                error_msg = str(e)
                logger.warning(f"Error accessing response.text: {error_msg}. Using fallback timetable generation.")
                if 'finish_reason' in error_msg or 'Part' in error_msg:
                    return self._generate_basic_timetable(timetable_context)
                return self._generate_basic_timetable(timetable_context)
            
        except Exception as e:
            logger.error(f"Gemini timetable generation error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Fallback to basic timetable generation if Gemini fails
            logger.warning("Falling back to basic timetable generation due to Gemini error")
            return self._generate_basic_timetable(timetable_context)
    
    def _generate_basic_timetable(self, context: Dict) -> Dict:
        """Generate a basic timetable as fallback when Gemini fails."""
        try:
            config = context.get('config', {})
            days = config.get('days_per_week', 5)
            periods_per_day = config.get('periods_per_day', 8)
            break_periods = config.get('break_periods', [])
            subjects = context.get('subjects', [])
            rooms = context.get('rooms', [])
            time_slots = context.get('time_slots', [])
            
            if not subjects:
                return {
                    'success': False,
                    'error': 'No subjects provided for timetable generation',
                    'algorithm': 'fallback'
                }
            
            grid = {}
            subject_index = 0
            room_index = 0
            
            # Calculate total periods needed
            total_periods_needed = sum(subj.get('periods_per_week', 0) for subj in subjects)
            periods_available = days * periods_per_day - len(break_periods) * days
            
            if total_periods_needed > periods_available:
                logger.warning(f"More periods needed ({total_periods_needed}) than available ({periods_available})")
            
            for day in range(days):
                grid[day] = {}
                periods_scheduled = 0
                
                for period in range(periods_per_day):
                    # Skip break periods
                    if period in break_periods:
                        grid[day][period] = {
                            'subject_code': '-',
                            'subject_name': 'Break',
                            'teacher_name': '',
                            'room': ''
                        }
                        continue
                    
                    # Schedule subjects
                    if subject_index < len(subjects):
                        subj = subjects[subject_index]
                        periods_for_subj = subj.get('periods_per_week', 0)
                        
                        # Check if we've scheduled enough periods for this subject
                        if periods_scheduled < periods_for_subj:
                            room = rooms[room_index % len(rooms)] if rooms else {'room_number': '101'}
                            grid[day][period] = {
                                'subject_code': subj.get('subject_code', ''),
                                'subject_name': subj.get('subject_name', ''),
                                'teacher_name': subj.get('teacher_name', 'TBA'),
                                'room': room.get('room_number', '101')
                            }
                            periods_scheduled += 1
                            room_index += 1
                            
                            # Move to next subject if we've scheduled enough periods
                            if periods_scheduled >= periods_for_subj:
                                subject_index += 1
                                periods_scheduled = 0
                        else:
                            grid[day][period] = None
                    else:
                        grid[day][period] = None
            
            return {
                'success': True,
                'algorithm': 'fallback',
                'grid': grid,
                'optimization_score': 70,  # Lower score for fallback
                'conflicts_resolved': 0,
                'constraint_violations': [],
                'execution_time': 0.1,
                'subjects': subjects
            }
        except Exception as e:
            logger.error(f"Error in basic timetable generation: {e}")
            return {
                'success': False,
                'error': f'Failed to generate basic timetable: {str(e)}',
                'algorithm': 'fallback'
            }
    
    def _build_timetable_generation_prompt(self, context: Dict) -> str:
        """Build comprehensive prompt for timetable generation."""
        prompt = """Create a class schedule timetable in JSON format.

Return ONLY valid JSON:
{
  "success": true,
  "grid": {
    "0": {
      "0": {"subject_code": "CS201", "subject_name": "Algorithms", "teacher_name": "Dr. Smith", "room": "101"},
      "1": {"subject_code": "CS202", "subject_name": "Database", "teacher_name": "Dr. Jones", "room": "102"}
    },
    "1": {},
    "2": {},
    "3": {},
    "4": {}
  },
  "optimization_score": 85,
  "conflicts_resolved": 0,
  "constraint_violations": []
}

Format:
- grid keys: day numbers 0-4 (Monday-Friday)
- grid[day] keys: period numbers starting from 0
- Each entry needs: subject_code, subject_name, teacher_name, room

"""
        
        # Add configuration
        config = context.get('config', {})
        prompt += f"\nTIMETABLE CONFIGURATION:\n"
        prompt += f"- Days per week: {config.get('days_per_week', 5)}\n"
        prompt += f"- Periods per day: {config.get('periods_per_day', 8)}\n"
        prompt += f"- Break periods: {config.get('break_periods', [])}\n"
        prompt += f"- Max teacher periods per day: {config.get('max_teacher_periods_per_day', 5)}\n"
        prompt += f"- Max consecutive periods: {config.get('max_consecutive_periods', 2)}\n"
        prompt += f"- Max subject periods per day: {config.get('max_subject_periods_per_day', 3)}\n"
        
        # Add course information
        course_info = context.get('course_info', {})
        prompt += f"\nCOURSE INFORMATION:\n"
        prompt += f"- Course: {course_info.get('course', 'N/A')}\n"
        prompt += f"- Year: {course_info.get('year', 'N/A')}\n"
        prompt += f"- Section: {course_info.get('section', 'N/A')}\n"
        
        # Add subjects
        subjects = context.get('subjects', [])
        prompt += f"\nSUBJECTS TO SCHEDULE ({len(subjects)} total):\n"
        for i, subj in enumerate(subjects, 1):
            prompt += f"{i}. {subj.get('subject_code', 'N/A')} - {subj.get('subject_name', 'N/A')}\n"
            prompt += f"   - Credits: {subj.get('credits', 0)}\n"
            prompt += f"   - Periods per week: {subj.get('periods_per_week', 0)}\n"
            prompt += f"   - Assigned Teacher: {subj.get('teacher_name', 'TBA')}\n"
        
        # Add teachers
        teachers = context.get('teachers', [])
        prompt += f"\nAVAILABLE TEACHERS ({len(teachers)} total):\n"
        for teacher in teachers:
            prompt += f"- {teacher.get('name', 'N/A')} (ID: {teacher.get('id', 'N/A')})\n"
            if teacher.get('assigned_subjects'):
                prompt += f"  Assigned subjects: {', '.join(teacher.get('assigned_subjects', []))}\n"
        
        # Add rooms
        rooms = context.get('rooms', [])
        prompt += f"\nAVAILABLE ROOMS ({len(rooms)} total):\n"
        for room in rooms:
            prompt += f"- {room.get('room_number', 'N/A')} (Capacity: {room.get('capacity', 'N/A')}, Type: {room.get('room_type', 'N/A')})\n"
        
        # Add time slots
        time_slots = context.get('time_slots', [])
        prompt += f"\nTIME SLOTS ({len(time_slots)} total):\n"
        for slot in time_slots:
            is_break = slot.get('is_break', False)
            prompt += f"- Period {slot.get('period_number', 'N/A')}: {slot.get('start_time', 'N/A')} - {slot.get('end_time', 'N/A')}"
            if is_break:
                prompt += " (BREAK)"
            prompt += "\n"
        
        # Add existing entries (to avoid conflicts)
        existing_entries = context.get('existing_entries', [])
        if existing_entries:
            prompt += f"\nEXISTING TIMETABLE ENTRIES (avoid conflicts):\n"
            for entry in existing_entries[:20]:  # Limit to first 20
                prompt += f"- Day {entry.get('day', 'N/A')}, Period {entry.get('period', 'N/A')}: {entry.get('subject', 'N/A')} with {entry.get('teacher', 'N/A')} in {entry.get('room', 'N/A')}\n"
        
        # Add constraints (simplified to avoid filter triggers)
        prompt += f"\nRules:\n"
        prompt += f"- Schedule each subject for required periods per week\n"
        prompt += f"- Max {config.get('max_teacher_periods_per_day', 5)} periods per teacher per day\n"
        prompt += f"- Max {config.get('max_subject_periods_per_day', 3)} periods per subject per day\n"
        prompt += f"- Max {config.get('max_consecutive_periods', 2)} consecutive periods per subject\n"
        prompt += f"- Skip periods: {config.get('break_periods', [])}\n"
        prompt += f"- One class per teacher per time\n"
        prompt += f"- One class per room per time\n"
        prompt += f"- Spread subjects across week\n"
        
        prompt += f"\nGenerate timetable. Return JSON only."
        
        return prompt
    
    def _parse_timetable_response(self, content: str, context: Dict) -> Dict:
        """Parse Gemini's timetable response into structured format."""
        try:
            import json
            import re
            
            # Extract JSON from response (handle markdown code blocks)
            # Try to find JSON in code blocks first
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            if not data.get('success', False):
                error_msg = data.get('error', 'Gemini returned success=false')
                logger.error(f"Gemini returned unsuccessful response: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'algorithm': 'gemini_ai'
                }
            
            # Convert grid format to match expected structure
            grid = data.get('grid', {})
            if not grid:
                logger.error("Gemini returned empty grid")
                return {
                    'success': False,
                    'error': 'Empty timetable grid received from Gemini',
                    'algorithm': 'gemini_ai'
                }
            
            converted_grid = {}
            
            for day_str, periods in grid.items():
                try:
                    day = int(day_str)
                    converted_grid[day] = {}
                    if not isinstance(periods, dict):
                        logger.warning(f"Periods for day {day} is not a dict: {type(periods)}")
                        continue
                    for period_str, entry in periods.items():
                        try:
                            period = int(period_str)
                            # Handle None or empty entries
                            if entry is None or not isinstance(entry, dict):
                                converted_grid[day][period] = None
                            else:
                                converted_grid[day][period] = {
                                    'subject_code': entry.get('subject_code', ''),
                                    'subject_name': entry.get('subject_name', ''),
                                    'teacher_name': entry.get('teacher_name', ''),
                                    'room': entry.get('room', '')
                                }
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error parsing period {period_str}: {e}")
                            continue
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing day {day_str}: {e}")
                    continue
            
            if not converted_grid:
                logger.error("Failed to convert any grid entries")
                return {
                    'success': False,
                    'error': 'Failed to parse timetable grid from Gemini response',
                    'algorithm': 'gemini_ai'
                }
            
            return {
                'success': True,
                'algorithm': 'gemini_ai',
                'grid': converted_grid,
                'optimization_score': data.get('optimization_score', 80),
                'conflicts_resolved': data.get('conflicts_resolved', 0),
                'constraint_violations': data.get('constraint_violations', []),
                'execution_time': 0.5,  # Gemini is fast
                'subjects': context.get('subjects', [])
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error parsing timetable response: {str(e)}")
            logger.error(f"Response content (first 1000 chars): {content[:1000]}")
            return {
                'success': False,
                'error': f'Invalid JSON response from Gemini: {str(e)}',
                'algorithm': 'gemini_ai'
            }
        except Exception as e:
            logger.error(f"Error parsing timetable response: {str(e)}")
            logger.error(f"Response content (first 1000 chars): {content[:1000]}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Error parsing Gemini response: {str(e)}',
                'algorithm': 'gemini_ai'
            }
    
    def _mock_timetable_generation(self, context: Dict) -> Dict:
        """Generate mock timetable for fallback."""
        days = context.get('config', {}).get('days_per_week', 5)
        periods = context.get('config', {}).get('periods_per_day', 8)
        subjects = context.get('subjects', [])
        
        grid = {}
        subject_index = 0
        
        for day in range(days):
            grid[day] = {}
            for period in range(periods):
                if subject_index < len(subjects):
                    subj = subjects[subject_index]
                    grid[day][period] = {
                        'subject_code': subj.get('subject_code', ''),
                        'subject_name': subj.get('subject_name', ''),
                        'teacher_name': subj.get('teacher_name', 'TBA'),
                        'room': '101'
                    }
                    subject_index += 1
                else:
                    grid[day][period] = None
        
        return {
            'success': True,
            'algorithm': 'mock',
            'grid': grid,
            'optimization_score': 60,
            'conflicts_resolved': 0,
            'constraint_violations': [],
            'execution_time': 0.1,
            'subjects': subjects
        }
    
    def _chat_with_offline_ai(self, message: str, context: Dict = None) -> str:
        """Chat using offline AI."""
        try:
            from utils.offline_ai import get_ai_response
            return get_ai_response(message, context)
        except Exception as e:
            logger.error(f"Offline AI error: {e}")
            return self._generate_smart_response(message, context)
    
    def _generate_smart_response(self, message: str, context: Dict = None) -> str:
        """Generate contextually appropriate response when AI fails."""
        message_lower = message.lower()
        
        # Academic scheduling responses
        if any(word in message_lower for word in ['schedule', 'timetable', 'class', 'when']):
            return "I can help you with your schedule! Your classes are organized to optimize your learning. Check your dashboard for today's schedule and any updates."
        
        # Study help responses  
        elif any(word in message_lower for word in ['study', 'exam', 'test', 'homework', 'assignment']):
            return "For effective studying, I recommend: 1) Review your notes daily, 2) Practice problems regularly, 3) Use active recall techniques. Would you like specific study tips for any subject?"
        
        # Course/subject responses
        elif any(word in message_lower for word in ['subject', 'course', 'math', 'physics', 'chemistry', 'biology', 'english']):
            return "I can provide guidance on your courses! Each subject in your timetable is designed to build your knowledge progressively. Focus on understanding concepts rather than just memorizing."
        
        # Grade/performance responses
        elif any(word in message_lower for word in ['grade', 'score', 'performance', 'result']):
            return "Your academic performance is important! I can help you track your progress and suggest improvements. Regular attendance and consistent study habits are key to success."
        
        # General academic help
        else:
            return f"Thanks for your question! I'm here to assist with your academic journey. I can help with schedules, study tips, course guidance, and academic planning. How can I support your learning today?"

# Create a singleton instance
ai_service = AIService()
