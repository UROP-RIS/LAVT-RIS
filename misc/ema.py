import torch
from collections import OrderedDict

@torch.no_grad()
def update_teacher_model(student, teacher, keep_rate=0.9996):
    """
    EMA update for teacher model.
    
    Args:
        student: Student model (can be DDP or plain)
        teacher: Teacher model (plain, not DDP)
        keep_rate: EMA momentum (0.99 ~ 0.9999)
    """
    # 获取 student 的 state_dict，兼容 DDP
    if hasattr(student, 'module'):
        student_state_dict = student.module.state_dict()
    else:
        student_state_dict = student.state_dict()

    teacher_state_dict = teacher.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in teacher_state_dict.items():
        if key not in student_state_dict:
            raise KeyError(f"Key {key} not found in student model. Teacher and student must have the same architecture.")
        
        # EMA update
        student_value = student_state_dict[key]
        updated_value = (1 - keep_rate) * student_value + keep_rate * value
        new_teacher_dict[key] = updated_value

    teacher.load_state_dict(new_teacher_dict)
