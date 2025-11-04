import { StateCreator } from 'zustand';

import { updateConversation } from '@/lib/firestore';
import { authSlice } from '@/lib/store/agent/slices/authSlice';
import { conversationSlice } from '@/lib/store/agent/slices/conversationSlice';
import { Message } from '@/types/chat/types';

const generateChatTitle = (firstMessage: string): string => {
  const words = firstMessage.split(" ").slice(0, 6).join(" ");
  return words.length > 40 ? words.substring(0, 40) + "..." : words;
};

export interface messageSlice {
  messages: Message[];
  input: string;
  status: "ready" | "loading" | "submitted" | "streaming" | "error";

  setInput: (input: string) => void;
  sendMessage: (chatId: string, mode: string) => Promise<void>;
  fetchMessages: (chatId: string) => Promise<void>;
  handleQuickAction: (action: string, chatId: string, mode: string) => void;
}

export const createMessageSlice: StateCreator<
  messageSlice & conversationSlice & authSlice,
  [],
  [],
  messageSlice
> = (set, get) => ({
  input: "",
  messages: [],
  status: "ready",

  setInput: (input) => set({ input }),

  fetchMessages: async (chatId) => {
    set({
      status: 'loading',
      messages: []
    });
    try {
      const chat = get().chats.find(
        c => c.id === chatId
      );
      if (!chat) {
        throw new Error("No messages found for this conversation.");
      }
      set({ messages: chat.messages || [], status: 'ready' });
    } catch (err) {
      console.error("Failed to fetch messages:", err);
      set({
        status: 'error',
        messages: []
      });
      throw err; // Re-throw the error
    }
  },

  sendMessage: async (chatId, mode) => {
    const messageContent = get().input.trim();
    if (!messageContent) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: messageContent,
      timestamp: Date.now(),
    };

    const isFirstMessage = get().messages.length === 0;
    const newTitle = isFirstMessage ? generateChatTitle(userMessage.content) : undefined;

    // Optimistic update with user message
    set((state) => {
      const updatedMessages = [...state.messages, userMessage];
      const updatedChats = state.chats
        .map(c =>
          c.id === chatId
            ? { ...c, messages: updatedMessages, updatedAt: new Date(), ...(newTitle && { title: newTitle }) } :
            c
        )
        .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());

      return { messages: updatedMessages, input: "", status: "submitted", chats: updatedChats };
    });

    try {
      const currentUser = get().currentUser;
      if (!currentUser) {
        throw new Error("User not Authenticated.");
      }

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [userMessage],
          mode,
          conversationId: chatId,
          userId: currentUser.uid,
        }),
      });

      if (!response.body) throw new Error("No response body from API.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let assistantResponse = "";
      const assistantMessageId = crypto.randomUUID();

      // Add empty assistant message placeholder + switch to streaming mode
      set(state => ({
        status: "streaming",
        messages: [...state.messages, {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          timestamp: Date.now()
        }]
      }));

      // Stream the response
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        assistantResponse += decoder.decode(value, { stream: true });

        set(state => ({
          messages: state.messages.map(msg =>
            msg.id === assistantMessageId ? { ...msg, content: assistantResponse } : msg
          )
        }));
      }

      // Done streaming → mark ready + update conversation and Firestore
      const finalMessages = get().messages;
      const finalDate = new Date();

      set(state => ({
        status: "ready",
        chats: state.chats
          .map(c =>
            c.id === chatId
              ? { ...c, messages: finalMessages, updatedAt: finalDate, ...(newTitle && { title: newTitle }) } :
              c
          )
          .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()),
      }));

      if (!currentUser) {
        throw new Error("User not authenticated.");
      }

      await updateConversation(chatId, currentUser.uid, {
        messages: finalMessages,
        updatedAt: finalDate,
        ...(newTitle && { title: newTitle }),
      });

    } catch (err) {
      console.error(`Error sending ${mode} message:`, err);
      set({ status: "error" });
      throw err; // Re-throw the error
    }
  },

  handleQuickAction: (action, chatId, mode) => {
    get().setInput(action);
    get().sendMessage(chatId, mode);
  },
});
