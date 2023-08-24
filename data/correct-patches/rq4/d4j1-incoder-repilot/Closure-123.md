# Repilot Patch

```
        Context rhsContext = getContextForNoInOperator(context);
```

# Developer Patch

```
        Context rhsContext = getContextForNoInOperator(context);
```

# Context

```
--- bug/Closure-123/src/com/google/javascript/jscomp/CodeGenerator.java

+++ fix/Closure-123/src/com/google/javascript/jscomp/CodeGenerator.java

@@ -282,7 +282,7 @@

       case Token.HOOK: {
         Preconditions.checkState(childCount == 3);
         int p = NodeUtil.precedence(type);
-        Context rhsContext = Context.OTHER;
+        Context rhsContext = getContextForNoInOperator(context);
         addExpr(first, p + 1, context);
         cc.addOp("?", true);
         addExpr(first.getNext(), 1, rhsContext);
```

# Note

