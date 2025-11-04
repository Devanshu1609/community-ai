import {
  addDoc, collection, deleteDoc,
  doc as firestoreDoc, getDoc, getDocs,
  query, updateDoc, where,
} from 'firebase/firestore';

import { db } from '@/app/firebase/config';
import type { ChatHistoryItem, Message } from '@/types/chat/types';

// Fetch all chats for a user, optionally filtered by mode.
export const fetchChats = async (
	userId: string,
	mode: string
): Promise<ChatHistoryItem[]> => {
	try {
		if (!userId) throw new Error('Unable to fetch chats.');

		const chatsCol = collection(db, 'chats');
		const q = query(
			chatsCol,
			where('userId', '==', userId),
			where('mode', '==', mode)
		);

		const chatsSnapshot = await getDocs(q);

		return chatsSnapshot.docs.map((d) => {
			const data = d.data();
			return {
				id: d.id,
				...data,
				createdAt: data.createdAt.toDate(),
				updatedAt: data.updatedAt?.toDate(),
			} as ChatHistoryItem;
		});
	} catch (error) {
		console.error('Error fetching chats:', error);
		throw new Error(
			error instanceof Error
				? `Failed to fetch chats: ${error.message}`
				: 'An unexpected error occurred while fetching chats.'
		);
	}
};

// Adds a new chat conversation.
export const addConversation = async (
	newChatData: Omit<ChatHistoryItem, 'id'>
): Promise<string> => {
	try {
		if (!newChatData.userId)
			throw new Error('Unable to create a conversation.');

		const docRef = await addDoc(collection(db, 'chats'), newChatData);
		return docRef.id;
	} catch (error) {
		console.error('Error adding new conversation:', error);
		throw new Error(
			error instanceof Error
				? `Failed to create conversation: ${error.message}`
				: 'An unexpected error occurred while creating a new conversation.'
		);
	}
};


// Updates an existing conversation, verifying ownership first.
export const updateConversation = async (
	conversationId: string,
	userId: string,
	updatedData: Partial<{ messages: Message[]; title: string; updatedAt: Date }>
) => {
	try {
		if (!conversationId) throw new Error('Missing conversation ID.');
		if (!userId) throw new Error('Missing user ID.');

		const chatRef = firestoreDoc(db, 'chats', conversationId);
		const chatSnap = await getDoc(chatRef);

		if (!chatSnap.exists()) {
			throw new Error('Chat not found. It may have been deleted.');
		}

		const chatData = chatSnap.data();
		if (chatData?.userId !== userId) {
			throw new Error('Access denied. You are not authorized to modify this chat.');
		}

		await updateDoc(chatRef, updatedData);
	} catch (error) {
		console.error(`Error updating chat ${conversationId}:`, error);
		throw new Error(
			error instanceof Error
				? `Failed to update chat: ${error.message}`
				: 'An unexpected error occurred while updating the chat.'
		);
	}
};


// Deletes a conversation, ensuring only the owner can delete it.
export const deleteConversationFromDB = async (
	conversationId: string,
	userId: string
) => {
	try {
		if (!conversationId) throw new Error('Missing conversation ID.');
		if (!userId) throw new Error('Missing user ID.');

		const chatRef = firestoreDoc(db, 'chats', conversationId);
		const chatSnap = await getDoc(chatRef);

		if (!chatSnap.exists()) {
			throw new Error('Chat not found. It may have already been deleted.');
		}

		const chatData = chatSnap.data();
		if (chatData?.userId !== userId) {
			throw new Error('Access denied. You can only delete your own chats.');
		}

		await deleteDoc(chatRef);
	} catch (error) {
		console.error(`Error deleting chat ${conversationId}:`, error);
		throw new Error(
			error instanceof Error
				? `Failed to delete chat: ${error.message}`
				: 'An unexpected error occurred while deleting the chat.'
		);
	}
};