# Repilot Patch

```
                    case '/':
                        out.write('\\');
                        out.write('/');
                        break;
```

# Developer Patch

```
                    case '/':
                        out.write('\\');
                        out.write('/');
                        break;
```

# Context

```
--- bug/Lang-52/src/java/org/apache/commons/lang/StringEscapeUtils.java

+++ fix/Lang-52/src/java/org/apache/commons/lang/StringEscapeUtils.java

@@ -232,6 +232,10 @@

                     case '\\':
                         out.write('\\');
                         out.write('\\');
+                        break;
+                    case '/':
+                        out.write('\\');
+                        out.write('/');
                         break;
                     default :
                         out.write(ch);
```

# Note

