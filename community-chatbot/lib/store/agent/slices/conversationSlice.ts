import { User } from 'firebase/auth';
import type { StateCreator } from 'zustand';

import { integrationModes } from '@/lib/constants/chat';
import {
  addConversation, deleteConversationFromDB, fetchChats,
  updateConversation,
} from '@/lib/firestore';
import { authSlice } from '@/lib/store/agent/slices/authSlice';
import { ChatHistoryItem } from '@/types/chat/types';

export interface conversationSlice {
    chats: ChatHistoryItem[];
    isCreatingConversation: boolean;
    pendingOperations: string[];
    conversationsLoaded: boolean; // Add this line

    fetchConversations: (currentUser: User, currentMode: string) => Promise<void>;
    createConversation: (currentMode: string) => Promise<string>;
    deleteConversation: (chatId: string) => Promise<void>;
    renameConversation: (chatId: string, newTitle: string) => Promise<void>;
    updateChatTimestamp: (chatId: string) => Promise<void>;
}

export const createConversationSlice: StateCreator<
    conversationSlice & authSlice,
    [],
    [],
    conversationSlice
> = (set, get) => ({
    chats: [], // List of all conv for the current mode
    isCreatingConversation: false, // True when a new chat is being created
    pendingOperations: [], // List of chat IDs currently being processed (delete/rename)
    conversationsLoaded: false, // Messages of a conv are not fetched before that conv

    fetchConversations: async (currentUser: User, currentMode: string) => {
        try {
            // Fetch all chats for current user and selected mode
            const conversations = await fetchChats(currentUser.uid, currentMode);
            // Sort chats by updatedAt (most recent first)
            const sortedConversations = conversations.sort(
                (a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()
            );
            set({
                chats: sortedConversations,
                conversationsLoaded: true // Set to true on success
            });
        } catch (error) {
            console.error("Error fetching conversations:", error);
            set({ conversationsLoaded: true }); // Also set to true on error to avoid infinite loading state
            throw error;
        }
    },

    createConversation: async (currentMode: string) => {
        // show loading in UI
        set({ isCreatingConversation: true });

        try {
            const { currentUser } = get();
            if (!currentUser) {
                throw new Error('User not Authenticated');
            }
            // Find metadata for current mode
            const modeDetails = integrationModes.find(
                (m) => m.id === currentMode
            );
            const newDate = new Date();
            // Create a new chat object (without ID yet)
            const newChatData: Omit<ChatHistoryItem, 'id'> = {
                title: `New ${modeDetails?.name || 'Chat'}`,
                createdAt: newDate,
                updatedAt: newDate,
                icon: modeDetails?.image || '/mifos.png',
                messages: [],
                mode: currentMode,
                userId: currentUser.uid,
            };

            const newId = await addConversation(newChatData);
            const newChat = { ...newChatData, id: newId };
            // Prepend new chat to local state array
            set((state) => ({
                chats: [newChat, ...state.chats] // Add at start
            }));
            return newId;
        } catch (error) {
            console.error('Failed to create new conversation:', error);
            throw error;
        } finally {
            set({ isCreatingConversation: false });
        }
    },

    deleteConversation: async (chatId: string) => {
        const { currentUser } = get();
        if (!currentUser) throw new Error("User not authenticated.");

        // Mark as pending
        set(state => ({
            pendingOperations: [...state.pendingOperations, chatId]
        }));

        try {
            await deleteConversationFromDB(chatId, currentUser.uid);
            set(state => ({
                chats: state.chats.filter(chat => chat.id !== chatId)
            }));
        } catch (error) {
            console.error("Error deleting conversation:", error);
            throw error;
        } finally {
            // Remove from pending whether success or error
            set(state => ({
                pendingOperations: state.pendingOperations.filter(id => id !== chatId)
            }));
        }
    },


    renameConversation: async (chatId: string, newTitle: string) => {
        const { currentUser } = get();
        if (!currentUser) throw new Error("User not Authenticated.");

        set(state => ({
            pendingOperations: [...state.pendingOperations, chatId]
        }));

        try {
            const newDate = new Date();
            set(state => ({
                chats: state.chats.map(chat =>
                    chat.id === chatId ? { ...chat, title: newTitle, updatedAt: newDate } : chat
                )
            }));
            await updateConversation(chatId, currentUser.uid, { title: newTitle, updatedAt: newDate });
        } catch (error) {
            console.error("Error renaming conversation:", error);
            throw error;
        } finally {
            set(state => ({
                pendingOperations: state.pendingOperations.filter(id => id !== chatId)
            }));
        }
    },

    updateChatTimestamp: async (chatId: string) => {
        const { currentUser } = get();
        if (!currentUser) {
            throw new Error("User not authenticated.");
        }

        try {
            const newDate = new Date();

            await updateConversation(chatId, currentUser.uid, { updatedAt: newDate });

            // Update timestamp locally to keep UI in sync
            set((state) => ({
                chats: state.chats.map((chat) =>
                    chat.id === chatId
                        ? { ...chat, updatedAt: newDate }
                        : chat
                ),
            }));
        } catch (error) {
            console.error("Error updating chat timestamp:", error);
            throw error; // Re-throw error for caller to handle
        }
    }
});
