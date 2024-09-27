import time
from functools import wraps
from enum import Enum

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"=============={func.__name__} took {end_time - start_time:.4f} seconds==============")
        return result
    return wrapper

class EntityType(str, Enum):
    # Compensation
    COMPENSATION = "compensation"
    COMPENSATION_SALARY = "compensation_salary"
    COMPENSATION_ALLOWANCE = "compensation_allowance"
    COMPENSATION_BONUS = "compensation_bonus"
    COMPENSATION_BENEFITS = "compensation_benefits"
    
    # Performance Management
    PERFORMANCE = "performance"
    PERFORMANCE_REVIEW = "performance_review"
    PERFORMANCE_GOAL = "performance_goal"
    PERFORMANCE_KPI = "performance_kpi"
    
    # Time Management
    TIME_MANAGEMENT = "time_management"
    TIME_ATTENDANCE = "time_attendance"
    TIME_LEAVE = "time_leave"
    TIME_OVERTIME = "time_overtime"
    
    # Career Development
    CAREER_DEVELOPMENT = "career_development"
    CAREER_PROMOTION = "career_promotion"
    CAREER_TRAINING = "career_training"
    CAREER_SKILLSET = "career_skillset"
    
    # Recruitment
    RECRUITMENT = "recruitment"
    RECRUITMENT_JOB_POSTING = "recruitment_job_posting"
    RECRUITMENT_INTERVIEW = "recruitment_interview"
    RECRUITMENT_ONBOARDING = "recruitment_onboarding"
    
    # Employee Relations
    EMPLOYEE_RELATIONS = "employee_relations"
    EMPLOYEE_RELATIONS_CONFLICT = "employee_relations_conflict"
    EMPLOYEE_RELATIONS_COMMUNICATION = "employee_relations_communication"
    EMPLOYEE_RELATIONS_ENGAGEMENT = "employee_relations_engagement"
    
    # Organization Structure
    ORGANIZATION = "organization"
    ORGANIZATION_DEPARTMENT = "organization_department"
    ORGANIZATION_POSITION = "organization_position"
    ORGANIZATION_REPORTING_LINE = "organization_reporting_line"
    ORGANIZATION_POSITION_LEVEL = "organization_position_level"
    ORGANIZATION_POSITION_DEPARTMENT = "organization_position_department"

    # Compliance and Legal
    COMPLIANCE = "compliance"
    COMPLIANCE_POLICY = "compliance_policy"
    COMPLIANCE_REGULATION = "compliance_regulation"
    COMPLIANCE_ETHICS = "compliance_ethics"
    
    # Health and Safety
    HEALTH_SAFETY = "health_safety"
    HEALTH_SAFETY_WORKPLACE = "health_safety_workplace"
    HEALTH_SAFETY_WELLNESS = "health_safety_wellness"
    
    # Offboarding
    OFFBOARDING = "offboarding"
    OFFBOARDING_RESIGNATION = "offboarding_resignation"
    OFFBOARDING_TERMINATION = "offboarding_termination"
    OFFBOARDING_EXIT_INTERVIEW = "offboarding_exit_interview"
    
    # System and Other
    SYSTEM = "system"
    OTHER = "other"

HR_ENTITY_LOCALIZATION = {
    EntityType.COMPENSATION: "薪酬相關",
    EntityType.COMPENSATION_SALARY: "薪資",
    EntityType.COMPENSATION_ALLOWANCE: "津貼",
    EntityType.COMPENSATION_BONUS: "獎金",
    EntityType.COMPENSATION_BENEFITS: "福利",
    
    EntityType.PERFORMANCE: "績效管理",
    EntityType.PERFORMANCE_REVIEW: "績效評估",
    EntityType.PERFORMANCE_GOAL: "目標設定",
    EntityType.PERFORMANCE_KPI: "關鍵績效指標",
    
    EntityType.TIME_MANAGEMENT: "時間管理",
    EntityType.TIME_ATTENDANCE: "出勤",
    EntityType.TIME_LEAVE: "請假",
    EntityType.TIME_OVERTIME: "加班",
    
    EntityType.CAREER_DEVELOPMENT: "職涯發展",
    EntityType.CAREER_PROMOTION: "晉升",
    EntityType.CAREER_TRAINING: "培訓",
    EntityType.CAREER_SKILLSET: "技能組合",
    
    EntityType.RECRUITMENT: "招聘",
    EntityType.RECRUITMENT_JOB_POSTING: "職位發布",
    EntityType.RECRUITMENT_INTERVIEW: "面試",
    EntityType.RECRUITMENT_ONBOARDING: "入職",
    
    EntityType.EMPLOYEE_RELATIONS: "員工關係",
    EntityType.EMPLOYEE_RELATIONS_CONFLICT: "衝突處理",
    EntityType.EMPLOYEE_RELATIONS_COMMUNICATION: "溝通",
    EntityType.EMPLOYEE_RELATIONS_ENGAGEMENT: "員工參與",
    
    EntityType.ORGANIZATION: "組織結構",
    EntityType.ORGANIZATION_DEPARTMENT: "部門",
    EntityType.ORGANIZATION_POSITION: "職位",
    EntityType.ORGANIZATION_REPORTING_LINE: "匯報線",
    EntityType.ORGANIZATION_POSITION_LEVEL: "職位等級",
    EntityType.ORGANIZATION_POSITION_DEPARTMENT: "職位所屬部門",

    EntityType.COMPLIANCE: "合規與法律",
    EntityType.COMPLIANCE_POLICY: "政策",
    EntityType.COMPLIANCE_REGULATION: "法規",
    EntityType.COMPLIANCE_ETHICS: "道德準則",
    
    EntityType.HEALTH_SAFETY: "健康與安全",
    EntityType.HEALTH_SAFETY_WORKPLACE: "工作場所安全",
    EntityType.HEALTH_SAFETY_WELLNESS: "健康計劃",
    
    EntityType.OFFBOARDING: "離職管理",
    EntityType.OFFBOARDING_RESIGNATION: "辭職",
    EntityType.OFFBOARDING_TERMINATION: "終止合約",
    EntityType.OFFBOARDING_EXIT_INTERVIEW: "離職面談",
    
    EntityType.SYSTEM: "系統",
    EntityType.OTHER: "其他"
}

class HRIntentCategory(str, Enum):
    # Queries
    QUERY = "query"
    QUERY_INFORMATION = "query_information"
    QUERY_POLICY = "query_policy"
    QUERY_STATUS = "query_status"
    
    # Requests
    REQUEST = "request"
    REQUEST_DOCUMENT = "request_document"
    REQUEST_APPROVAL = "request_approval"
    REQUEST_CHANGE = "request_change"
    
    # Reports
    REPORT = "report"
    REPORT_ISSUE = "report_issue"
    REPORT_PROGRESS = "report_progress"
    REPORT_RESULT = "report_result"
    
    # Processes
    PROCESS = "process"
    PROCESS_INITIATE = "process_initiate"
    PROCESS_UPDATE = "process_update"
    PROCESS_COMPLETE = "process_complete"
    
    # Feedback
    FEEDBACK = "feedback"
    FEEDBACK_PROVIDE = "feedback_provide"
    FEEDBACK_REQUEST = "feedback_request"
    
    # Assistance
    ASSISTANCE = "assistance"
    ASSISTANCE_GUIDANCE = "assistance_guidance"
    ASSISTANCE_CLARIFICATION = "assistance_clarification"
    ASSISTANCE_SUPPORT = "assistance_support"
    
    # Analysis
    ANALYSIS = "analysis"
    ANALYSIS_DATA = "analysis_data"
    ANALYSIS_TREND = "analysis_trend"
    ANALYSIS_PERFORMANCE = "analysis_performance"
    
    # Decision Support
    DECISION_SUPPORT = "decision_support"
    DECISION_SUPPORT_RECOMMENDATION = "decision_support_recommendation"
    DECISION_SUPPORT_OPTION = "decision_support_option"
    
    # Learning and Development
    LEARNING = "learning"
    LEARNING_COURSE = "learning_course"
    LEARNING_SKILL = "learning_skill"
    LEARNING_CAREER_PATH = "learning_career_path"
    
    # System Related
    SYSTEM = "system"
    SYSTEM_ACCESS = "system_access"
    SYSTEM_ISSUE = "system_issue"
    SYSTEM_ENHANCEMENT = "system_enhancement"
    
    # Other
    OTHER = "other"

HR_INTENT_LOCALIZATION = {
    HRIntentCategory.QUERY: "查詢",
    HRIntentCategory.QUERY_INFORMATION: "資訊查詢",
    HRIntentCategory.QUERY_POLICY: "政策諮詢",
    HRIntentCategory.QUERY_STATUS: "狀態查詢",
    
    HRIntentCategory.REQUEST: "請求",
    HRIntentCategory.REQUEST_DOCUMENT: "文件請求",
    HRIntentCategory.REQUEST_APPROVAL: "審批請求",
    HRIntentCategory.REQUEST_CHANGE: "變更請求",
    
    HRIntentCategory.REPORT: "報告",
    HRIntentCategory.REPORT_ISSUE: "問題報告",
    HRIntentCategory.REPORT_PROGRESS: "進度報告",
    HRIntentCategory.REPORT_RESULT: "結果報告",
    
    HRIntentCategory.PROCESS: "流程",
    HRIntentCategory.PROCESS_INITIATE: "流程啟動",
    HRIntentCategory.PROCESS_UPDATE: "流程更新",
    HRIntentCategory.PROCESS_COMPLETE: "流程完成",
    
    HRIntentCategory.FEEDBACK: "回饋",
    HRIntentCategory.FEEDBACK_PROVIDE: "提供回饋",
    HRIntentCategory.FEEDBACK_REQUEST: "請求回饋",
    
    HRIntentCategory.ASSISTANCE: "協助",
    HRIntentCategory.ASSISTANCE_GUIDANCE: "尋求指導",
    HRIntentCategory.ASSISTANCE_CLARIFICATION: "尋求澄清",
    HRIntentCategory.ASSISTANCE_SUPPORT: "尋求支持",
    
    HRIntentCategory.ANALYSIS: "分析",
    HRIntentCategory.ANALYSIS_DATA: "數據分析",
    HRIntentCategory.ANALYSIS_TREND: "趨勢分析",
    HRIntentCategory.ANALYSIS_PERFORMANCE: "績效分析",
    
    HRIntentCategory.DECISION_SUPPORT: "決策支持",
    HRIntentCategory.DECISION_SUPPORT_RECOMMENDATION: "建議",
    HRIntentCategory.DECISION_SUPPORT_OPTION: "選項評估",
    
    HRIntentCategory.LEARNING: "學習發展",
    HRIntentCategory.LEARNING_COURSE: "課程查詢",
    HRIntentCategory.LEARNING_SKILL: "技能發展",
    HRIntentCategory.LEARNING_CAREER_PATH: "職業路徑",
    
    HRIntentCategory.SYSTEM: "系統相關",
    HRIntentCategory.SYSTEM_ACCESS: "系統訪問",
    HRIntentCategory.SYSTEM_ISSUE: "系統問題",
    HRIntentCategory.SYSTEM_ENHANCEMENT: "系統改進建議",
    
    HRIntentCategory.OTHER: "其他"
}

def get_Chinese_intent(intent: HRIntentCategory) -> str:
    return HR_INTENT_LOCALIZATION.get(intent, str(intent))