"""
conversation_manager.py
=========================================
聊天数据管理
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


# ==========================================
# 一条消息
# ==========================================

@dataclass
class Message:

    role: str

    content: str

    time: datetime = field(
        default_factory=datetime.now
    )


# ==========================================
# 一个聊天
# ==========================================

@dataclass
class Conversation:

    title: str

    messages: List[Message] = field(
        default_factory=list
    )


# ==========================================
# 会话管理器
# ==========================================

class ConversationManager:

    def __init__(self):

        self.title = "新的聊天"

        self.messages = []

        self.conversations = []

        self.current_index = -1

    # ======================================

    def new_conversation(self):

        title = f"聊天 {len(self.conversations)+1}"

        conv = Conversation(title)

        self.conversations.append(conv)

        self.current_index = len(self.conversations)-1

        return self.current_index

    # ======================================

    def current(self):

        if self.current_index < 0:

            return None

        return self.conversations[
            self.current_index
        ]

    # ======================================

    def switch(self,index):

        if index < 0:

            return

        if index >= len(self.conversations):

            return

        self.current_index=index

    # ======================================

    def add_user_message(self,text):

        self.current().messages.append(

            Message(

                role="user",

                content=text

            )

        )

    # ======================================

    def add_bot_message(self,text):

        self.current().messages.append(

            Message(

                role="assistant",

                content=text

            )

        )

    # ======================================

    def get_messages(self):

        return self.current().messages

    # ======================================

    def conversation_titles(self):

        return [

            c.title

            for c in self.conversations

        ]

    # ======================================

    def rename_current(self,title):

        self.current().title=title

    # ======================================

    def remove(self,index):

        if len(self.conversations)<=1:

            return

        del self.conversations[index]

        self.current_index=min(

            self.current_index,

            len(self.conversations)-1

        )