# Repilot Patch

```
return compute(args[0].compute(context), args[1].compute(context))
```

# Developer Patch

```
        return compute(args[0].compute(context), args[1].compute(context))
```

# Context

```
--- bug/JxPath-10/src/java/org/apache/commons/jxpath/ri/compiler/CoreOperationRelationalExpression.java

+++ fix/JxPath-10/src/java/org/apache/commons/jxpath/ri/compiler/CoreOperationRelationalExpression.java

@@ -39,7 +39,7 @@

     }
 
     public final Object computeValue(EvalContext context) {
-        return compute(args[0].computeValue(context), args[1].computeValue(context)) 
+return compute(args[0].compute(context), args[1].compute(context))
                 ? Boolean.TRUE : Boolean.FALSE;
     }
```

# Note

