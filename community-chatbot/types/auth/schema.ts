import * as z from 'zod';

export const signUpSchema = z.object({
	name: z.string().min(1, "Name is required"),
	email: z.string().email("Invalid email address"),
	password: z.string().min(8, "Password must be at least 8 characters"),
	confirmPassword: z.string().min(1, "Please confirm your password"),
	acceptTerms: z.boolean().refine((val) => val === true, "You must accept the terms and conditions"),
}).refine((data) => data.password === data.confirmPassword, {
	message: "Passwords do not match",
	path: ["confirmPassword"],
})

export type SignUpSchema = z.infer<typeof signUpSchema>

export const signInSchema = z.object({
	email: z.string().email("Enter a valid email"),
	password: z.string().min(8, "Password must be at least 8 characters"),
})
export type SignInSchema = z.infer<typeof signInSchema>